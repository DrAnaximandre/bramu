# Bramu (Brain music)


## Description

Using an consumer-grade EEG device ([Muse 2](https://choosemuse.com/muse-2/)), and [Sonic Pi](https://sonic-pi.net/), one can produce interesting sounds.


This repo deals with:
- (GUI) connecting the EEG device to a Windows or Ubuntu computer;
- (Python) transforming voltages coming from the EEG device to brain waves through FFT;
- (Python) computing brain activity metrics;
- (Python) sending OSC messages on localhost;
- (Python) creating vizualisations 
- (Sonic Pi) receiving OSC messages and converting them to sounds.

## Dependencies

Python 3.7 +
- `numpy`
- `vispy` (advised to install via conda: `conda install -c conda-forge vispy`)
- a backend for `vispy` (recommended: `pyqt`, via conda)
- `pylsl`
- `python-osc`
- `seaborn`

Sonic Pi 3.1 +

## Contributing
Contributions and collaborations are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
Thanks given to the amazing neuro and Sonic Pi communities.

Thanks to @hyruuk for help connecting the Muse to a laptop with eegsynth.

The core Python code was adapted from https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/neurofeedback.py

## Which file should I use ? 

 - You should use `muse-windows.py` to send OSC.
 - `new-app.py` provides visualisation but does not have a calibration time and does not send OSC messages yet.

![alt text](new-vis.png "Visualisation of `new-app.py`")

## History

Bramu is an art project derived from an installation done at the [Montreal MusicMotion Hacklab 2019](https://musicmotion.org/hacklab-en), whithin the FundamentalFrequencies team. The installation featured an EEG device ([Muse 2](https://choosemuse.com/muse-2/)), a haptic device ([Subpac](https://subpac.com/)) and a visual shader that reacts to the participant's brain activity.

Real-time brain activity metrics were derived from the EEG signals using Python and sent in OSC format to the other parts of the installation (samely Sonic-Pi for the haptic and [Unity](https://unity.com/) for the shader).

## License
[MIT](https://choosealicense.com/licenses/mit/)