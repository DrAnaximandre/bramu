# -*- coding: utf-8 -*-
"""
Fundamental Frequency:

new version of the app so that I have something to monitor
while the experience plays.
"""

from pylsl import StreamInlet, resolve_byprop
from vispy import app
from pythonosc import udp_client
from seaborn import color_palette
import numpy as np


class Canvas(app.Canvas):
    """ docstring """

    def __init__(self,
                 lsl_inlet,
                 osc_sender):

        super(Canvas, self).__init__()

        self.inlet = lsl_inlet
        self.sender = osc_sender

        info = self.inlet.info()
        description = info.desc()

        window = 10
        self.sfreq = info.nominal_srate()
        n_samples = int(self.sfreq * window)

        n_rows = 1
        n_cols = 1

        # Number of signals.
        m = n_rows * n_cols
        # Number of samples per signal.
        n = n_samples

        # Various signal amplitudes.
        y = np.zeros((m, n)).astype(np.float32)


        color = color_palette("RdBu_r", n_rows)



        sys.exit()

    def send_osc_message(self, value, name):
        self.sender.send_message('/fund_freq/' + name, value)


def view():

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
    view()
