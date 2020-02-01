# -*- coding: utf-8 -*-
"""
Fundamental Frequency:

new version of the app so that I have something to monitor
while the experience plays.

Inspired by
https://github.com/alexandrebarachant/muse-lsl/blob/master/muselsl/viewer_v2.py
and
https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/neurofeedback.py

"""

from pylsl import StreamInlet, resolve_byprop
from vispy import app, gloo, visuals
from pythonosc import udp_client
from seaborn import color_palette
import numpy as np
from utils import update_buffer, get_last_data, compute_band_powers

VERT_SHADER = """
#version 120
// y coordinate of the position.
attribute float a_position;
// row, col, and time index.
attribute vec3 a_index;
varying vec3 v_index;
// 2D scaling factor (zooming).
uniform vec2 u_scale;
// Size of the table.
uniform vec2 u_size;
// Number of samples per signal.
uniform float u_n;
// Color.
attribute vec3 a_color;
varying vec4 v_color;
// Varying variables used for clipping in the fragment shader.
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    float n_rows = u_size.x;
    float n_cols = u_size.y;
    // Compute the x coordinate from the time index.
    float x = -1 + 2*a_index.z / (u_n-1);
    vec2 position = vec2(x - (1 - 1 / u_scale.x), a_position);
    // Find the affine transformation for the subplots.
    vec2 a = vec2(1./n_cols, 1./n_rows)*.9;
    vec2 b = vec2(-1 + 2*(a_index.x+.5) / n_cols,
                    -1 + 2*(a_index.y+.5) / n_rows);
    // Apply the static subplot transformation + scaling.
    gl_Position = vec4(a*u_scale*position+b, 0.0, 1.0);
    v_color = vec4(a_color, 1.);
    v_index = a_index;
    // For clipping test in the fragment shader.
    v_position = gl_Position.xy;
    v_ab = vec4(a, b);
}
"""

FRAG_SHADER = """
#version 120
varying vec4 v_color;
varying vec3 v_index;
varying vec2 v_position;
varying vec4 v_ab;
void main() {
    gl_FragColor = v_color;
    // Discard the fragments between the signals (emulate glMultiDrawArrays).
    if ((fract(v_index.x) > 0.) || (fract(v_index.y) > 0.))
        discard;
    // Clipping test.
    vec2 test = abs((v_position.xy-v_ab.zw)/v_ab.xy);
    if ((test.x > 1))
        discard;
}
"""

""" CALIBRATION PARAMETERS """
# Calibration time in seconds
CALIBRATION_TIME = 25

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 10

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 1 - 1 / 60

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0, 1, 2, 3]


class Canvas(app.Canvas):
    """ docstring """

    def __init__(self,
                 lsl_inlet,
                 osc_sender):

        app.Canvas.__init__(self, title='test',
                            keys='interactive')

        self.inlet = lsl_inlet
        self.sender = osc_sender

        # info = self.inlet.info()
        # description = info.desc()

        self.window = 5  # seconds
        self.sfreq = 256  # info.nominal_srate()
        self.n_samples = int(self.sfreq * self.window)
        self.n_chans = 4  # info.channel_count()

        self.n_rows = 5
        self.n_cols = 1

        # Number of signals.
        m = self.n_rows * self.n_cols
        # Number of samples per signal.
        n = self.n_samples

        # Various signal amplitudes.
        y = np.zeros((m, n)).astype(np.float32)

        color = color_palette("RdBu_r", self.n_rows)

        color = np.repeat(color, n, axis=0).astype(np.float32)

        # Signal 2D index of each vertex (row and col) and x-index (sample index
        # within each signal).
        index = np.c_[np.repeat(np.repeat(np.arange(self.n_cols), self.n_rows), n),
                      np.repeat(
                          np.tile(np.arange(self.n_rows), self.n_cols), n),
                      np.tile(np.arange(n), m)].astype(np.float32)

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (self.n_rows, self.n_cols)
        self.program['u_n'] = n

        self.filt = True
        self.af = [1.0]

        self.eeg_buffer = np.zeros((int(self.sfreq * self.window), 4))
        self.filter_state = None
        self.band_buffer = np.zeros((int(self.sfreq * self.window), 4))

        """ TEXTS """
        self.band_names = []  # displayed left
        band_names = ["Score", "Delta", "Theta", "Alpha", "Beta"]

        self.last_values = []  # displayed right, upper
        self.mean_values = []  # displayed right, lower

        for ii in range(self.n_rows):
            text = visuals.TextVisual(band_names[ii], bold=True, color='white')
            self.band_names.append(text)
            text = visuals.TextVisual('', bold=True, color='white')
            self.last_values.append(text)
            text = visuals.TextVisual('', bold=True, color='white')
            self.mean_values.append(text)

        self.quality_colors = color_palette("RdYlGn", 11)[::-1]

        """ APP INTERNALS  """
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))
        self.show()

    def on_timer(self, event):

        eeg_data, _ = self.inlet.pull_chunk(
            timeout=1, max_samples=int(SHIFT_LENGTH * self.sfreq))

        if eeg_data:

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            # Update EEG buffer with the new data
            self.eeg_buffer, self.filter_state = update_buffer(
                self.eeg_buffer, ch_data, notch=True,
                filter_state=self.filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = get_last_data(self.eeg_buffer, int(self.sfreq))

            # Compute band powers
            band_powers = compute_band_powers(data_epoch, self.sfreq)
            self.band_buffer, _ = update_buffer(self.band_buffer,
                                                np.asarray([band_powers]))

            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(self.band_buffer, axis=0)

            beta = self.band_buffer[:, 3]
            # add an epsilon to avoid 0 division ...
            alpha = self.band_buffer[:, 2] + 1e-8
            keep = int(self.window * self.sfreq)
            scores = np.reshape(beta / alpha, (keep, 1))

            smooth_score = np.mean(scores, axis=0)

            score_and_band_buffer = np.concatenate(
                (scores, self.band_buffer), axis=1)

            last = get_last_data(score_and_band_buffer, 1)
            print(last)

            self.band_names[0].font_size = 16
            self.last_values[0].text = 'last: %.2f' % (last[0, 0])
            self.mean_values[0].text = 'mean: %.2f' % (smooth_score)
            for i in range(self.n_chans):
                self.band_names[i + 1].font_size = 16
                self.last_values[i + 1].text = 'last: %.2f' % (last[0, i + 1])
                self.mean_values[i +
                                 1].text = 'mean: %.2f' % (smooth_band_powers[i])

            self.program['a_position'].set_data(
                score_and_band_buffer.T.ravel().astype(np.float32))
            self.update()

    def on_draw(self, event):
        gloo.clear()
        gloo.set_viewport(0, 0, *self.physical_size)
        self.program.draw('line_strip')
        [t.draw() for t in self.band_names + self.last_values + self.mean_values]

    def send_osc_message(self, value, name):
        self.sender.send_message('/fund_freq/' + name, value)

    def on_resize(self, event):
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)

        for ii, t in enumerate(self.band_names):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.035,
                     ((ii + 0.5) / self.n_rows) * self.size[1])

        for ii, t in enumerate(self.last_values):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.95,
                     ((ii + 0.45) / self.n_rows) * self.size[1])

        for ii, t in enumerate(self.mean_values):
            t.transforms.configure(canvas=self, viewport=vp)
            t.pos = (self.size[0] * 0.95,
                     ((ii + 0.55) / self.n_rows) * self.size[1])


def main():

    sender = udp_client.SimpleUDPClient('127.0.0.1', 4559)

    print("Looking for an EEG stream...")
    streams = resolve_byprop('type', 'EEG', timeout=10)

    if len(streams) == 0:
        raise(RuntimeError("Can't find EEG stream."))
    print("Start acquiring data.")

    inlet = StreamInlet(streams[0], max_chunklen=12)
    Canvas(inlet, sender)
    app.run()


if __name__ == '__main__':
    main()
