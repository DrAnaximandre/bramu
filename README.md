# Fundamental Frequencies


## Description

Fundamental Frequencies is an art project that was initiated at the [Montreal MusicMotion Hacklab 2019](https://musicmotion.org/hacklab-en).

The installation features an EEG device ([Muse 2](https://choosemuse.com/muse-2/)), a haptic device ([Subpac](https://subpac.com/)) and a visual shader that reacts to the participant's brain activity.

Real-time brain activity metrics are derived from the EEG signals using Python and sent in OSC format to the other parts of the installation (namely [Sonic Pi](https://sonic-pi.net/) for the haptic and [Unity](https://unity.com/) for the shader).

This repo deals with:
- (GUI) connecting the EEG device to a Windows or Ubuntu computer;
- (Python) transforming voltages to brain waves through FFT;
- (Python) computing brain activity metrics;
- (Python) sending OSC messages on localhost;
- (Sonic Pi) receiving OSC messages and converting them to low-frequency sounds.

## Dependencies

Python 3.7 +
- `numpy`
- `pylsl`
- `pythonosc`

Sonic Pi 3.1 +

## Contributing
Contributions and collaborations are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments
Thanks given to the amazing neuro and sonic-pi communities.

Thanks to @hyruuk for help connecting the Muse to a laptop with eegsynth.

The core Python code was adapted from https://github.com/alexandrebarachant/muse-lsl/blob/master/examples/neurofeedback.py

## License
[MIT](https://choosealicense.com/licenses/mit/)