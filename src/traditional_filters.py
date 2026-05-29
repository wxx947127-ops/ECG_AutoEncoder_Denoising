import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


def butterworth_lowpass_filter(signal, fs=500, cutoff=40, order=4):
    signal = np.asarray(signal, dtype=np.float32)
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal).astype(np.float32)


def notch_filter(signal, fs=500, notch_freq=50, quality_factor=30):
    signal = np.asarray(signal, dtype=np.float32)
    b, a = iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    return filtfilt(b, a, signal).astype(np.float32)


def median_filter(signal, kernel_size=5):
    signal = np.asarray(signal, dtype=np.float32)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(signal, kernel_size=kernel_size).astype(np.float32)


def combined_filter(signal, fs=500):
    filtered = notch_filter(signal, fs=fs)
    filtered = butterworth_lowpass_filter(filtered, fs=fs)
    filtered = median_filter(filtered, kernel_size=5)
    return np.clip(filtered, -1.0, 1.0).astype(np.float32)
