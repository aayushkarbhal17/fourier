"""
Unit tests for frequency analysis, single-sided FFT, and peak extraction.
"""

import numpy as np
import pytest
from src.frequency_analysis import compute_single_sided_fft, find_spectral_peaks
from src.signal_generator import generate_time_array, generate_ideal_sine, generate_ideal_square


def test_fft_pure_sine_amplitude_and_frequency():
    """
    A pure sine wave of amplitude A=2.5 at frequency f0=10 Hz
    must yield an FFT peak at exactly 10 Hz with amplitude ~2.5.
    """
    f0 = 10.0
    A = 2.5
    fs = 1000.0
    duration = 2.0  # Exactly 20 full cycles
    t = np.linspace(0.0, duration, int(fs * duration), endpoint=False)
    signal = A * np.sin(2.0 * np.pi * f0 * t)

    freqs, amps, meta = compute_single_sided_fft(signal, fs=fs, window_type="rectangular")

    # Find peak index
    peak_idx = np.argmax(amps)
    detected_freq = freqs[peak_idx]
    detected_amp = amps[peak_idx]

    assert pytest.approx(detected_freq, abs=0.5) == f0
    assert pytest.approx(detected_amp, rel=0.02) == A
    assert meta["nyquist_hz"] == 500.0


def test_fft_dc_offset_amplitude():
    """
    A signal with a DC bias of 3.0 must produce a DC bin (k=0) amplitude of 3.0.
    """
    fs = 500.0
    t = np.linspace(0.0, 1.0, 500, endpoint=False)
    signal = 3.0 + 1.0 * np.cos(2.0 * np.pi * 5.0 * t)

    freqs, amps, _ = compute_single_sided_fft(signal, fs=fs)
    assert pytest.approx(amps[0], rel=0.01) == 3.0


def test_find_spectral_peaks_square_wave():
    """
    For a square wave, spectral peaks must appear at odd multiples:
    f0, 3*f0, 5*f0...
    """
    f0 = 5.0
    t, fs = generate_time_array(f0=f0, num_periods=10.0, samples_per_period=200)
    signal = generate_ideal_square(t, f0=f0, amplitude=1.0)

    freqs, amps, _ = compute_single_sided_fft(signal, fs=fs)
    peaks_df = find_spectral_peaks(freqs, amps, f0=f0, amplitude_threshold=0.05, max_peaks=6)

    assert not peaks_df.empty
    detected_harmonics = list(peaks_df["Harmonic (n)"])
    
    # Harmonic 1, 3, 5 should be detected
    assert 1 in detected_harmonics
    assert 3 in detected_harmonics
    assert 5 in detected_harmonics

    # Even harmonics 2, 4 should NOT be in significant peaks
    assert 2 not in detected_harmonics
    assert 4 not in detected_harmonics


def test_fft_windowing_types():
    """
    Verifies that Hann, Hamming, and Blackman windows run without error
    and preserve main peak location.
    """
    f0 = 8.0
    t = np.linspace(0.0, 1.0, 1000, endpoint=False)
    signal = np.sin(2.0 * np.pi * f0 * t)

    for w_type in ["hann", "hamming", "blackman"]:
        freqs, amps, meta = compute_single_sided_fft(signal, fs=1000.0, window_type=w_type)
        peak_idx = np.argmax(amps)
        assert pytest.approx(freqs[peak_idx], abs=1.0) == f0
        assert meta["window_applied"] == w_type
