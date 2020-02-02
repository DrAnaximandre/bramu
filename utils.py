# -*- coding: utf-8 -*-
"""
Muse LSL Example Auxiliary Tools
These functions perform the lower-level operations involved in buffering,
epoching, and transforming EEG data into frequency bands
@author: Cassani
"""
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi


NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')


def sigmoid(x):
    """
        Returns the sigmoid of a value
    """
    return(1 / (1 + np.exp(x)))


def compute_band_powers(eegdata, fs):
    """Extract the features (band powers) from the EEG.
    Args:
        eegdata (numpy.ndarray): array of dimension [number of samples,
                number of channels]
        fs (float): sampling frequency of eegdata
    Returns:
        (numpy.ndarray): feature matrix of shape [number of feature points,
            number of different features]
    """
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape
    # Apply Hamming window
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])

    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(np.mean(PSD[ind_delta, :], axis=0))
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(np.mean(PSD[ind_theta, :], axis=0))
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(np.mean(PSD[ind_alpha, :], axis=0))
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(np.mean(PSD[ind_beta, :], axis=0))

    feature_vector = np.concatenate(([meanDelta], [meanTheta], [meanAlpha],
                                     [meanBeta]), axis=0)
    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            # Initialize feature_matrix
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))

        feature_matrix[i_epoch, :] = compute_band_powers(
            epochs[:, :, i_epoch], fs).T

    return feature_matrix


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """

    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer


def get_band_powers(inlet, eeg_buffer, filter_state, band_buffer,
                    SHIFT_LENGTH, INDEX_CHANNEL, EPOCH_LENGTH, fs):
    """ 3.1 ACQUIRE DATA """
    # Obtain EEG data from the LSL stream
    eeg_data, timestamp = inlet.pull_chunk(
        timeout=1, max_samples=int(SHIFT_LENGTH * fs))

    # Only keep the channel we're interested in
    ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
    # Update EEG buffer with the new data
    eeg_buffer, filter_state = update_buffer(
        eeg_buffer, ch_data, notch=True,
        filter_state=filter_state)

    """ 3.2 COMPUTE BAND POWERS """
    # Get newest samples from the buffer
    data_epoch = get_last_data(eeg_buffer,
                               EPOCH_LENGTH * fs)

    # Compute band powers
    band_powers = compute_band_powers(data_epoch, fs)
    band_buffer, _ = update_buffer(band_buffer,
                                   np.asarray([band_powers]))
    # Compute the average band powers for all epochs in buffer
    # This helps to smooth out noise
    smooth_band_powers = np.mean(band_buffer, axis=0)

    return smooth_band_powers, band_powers
