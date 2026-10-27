"""
Frequency Analysis and FFT Module for FourierLab
Provides single-sided amplitude spectrum calculation, spectral peak extraction,
windowing, and educational interpretations.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def compute_single_sided_fft(
    signal: np.ndarray,
    fs: float,
    window_type: str = "rectangular"
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Computes mathematically rigorous single-sided amplitude spectrum for real signals:
    - Uses rfft (real-valued fast Fourier transform)
    - Normalizes DC by 1/N and AC harmonics by 2/N
    - Handles windowing and coherent gain correction
    """
    n_samples = len(signal)
    if n_samples == 0:
        return np.array([]), np.array([]), {}

    # Window selection
    w_type = window_type.lower()
    if w_type == "hann":
        window = np.hanning(n_samples)
    elif w_type == "hamming":
        window = np.hamming(n_samples)
    elif w_type == "blackman":
        window = np.blackman(n_samples)
    else:
        window = np.ones(n_samples)
        w_type = "rectangular"

    # Coherent gain factor for amplitude preservation
    coherent_gain = np.mean(window)
    windowed_signal = signal * window

    # Fast Fourier Transform
    raw_fft = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Scale amplitudes: DC by 1/N, AC by 2/N
    amplitudes = (2.0 * np.abs(raw_fft)) / (n_samples * coherent_gain)
    amplitudes[0] /= 2.0  # Correct DC bin

    # If N is even, the last bin is Nyquist frequency and does not have a symmetric twin
    if n_samples % 2 == 0 and len(amplitudes) > 1:
        amplitudes[-1] /= 2.0

    meta = {
        "num_samples": n_samples,
        "sampling_rate_hz": fs,
        "freq_resolution_hz": fs / n_samples,
        "nyquist_hz": fs / 2.0,
        "duration_sec": n_samples / fs,
        "window_applied": w_type
    }

    return freqs, amplitudes, meta


def find_spectral_peaks(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    f0: float,
    amplitude_threshold: float = 0.02,
    max_peaks: int = 15
) -> pd.DataFrame:
    """
    Locates significant harmonic peaks and relates them to harmonic multiples n * f0.
    """
    if len(freqs) == 0:
        return pd.DataFrame()

    peak_rows = []
    f_res = freqs[1] - freqs[0] if len(freqs) > 1 else 0.1
    half_bin_search = max(1, int(round((0.35 * f0) / f_res)))

    # Look for multiples n = 1, 2, ..., up to Nyquist
    for n in range(1, max_peaks + 1):
        target_f = n * f0
        if target_f >= freqs[-1]:
            break

        idx_center = np.argmin(np.abs(freqs - target_f))
        idx_start = max(0, idx_center - half_bin_search)
        idx_end = min(len(freqs), idx_center + half_bin_search + 1)

        local_window_amps = amplitudes[idx_start:idx_end]
        if len(local_window_amps) == 0:
            continue

        local_max_idx = idx_start + int(np.argmax(local_window_amps))
        peak_amp = float(amplitudes[local_max_idx])
        peak_freq = float(freqs[local_max_idx])

        if peak_amp >= amplitude_threshold:
            peak_rows.append({
                "Harmonic (n)": n,
                "Nominal Freq (Hz)": round(target_f, 2),
                "Detected Freq (Hz)": round(peak_freq, 2),
                "Spectral Amplitude": round(peak_amp, 4),
                "Relative to Fund (%)": 0.0  # will calculate below
            })

    if not peak_rows:
        return pd.DataFrame(columns=["Harmonic (n)", "Nominal Freq (Hz)", "Detected Freq (Hz)", "Spectral Amplitude", "Relative to Fund (%)"])

    df = pd.DataFrame(peak_rows)
    fund_amp = df.iloc[0]["Spectral Amplitude"] if len(df) > 0 and df.iloc[0]["Harmonic (n)"] == 1 else 1.0
    if fund_amp > 1e-9:
        df["Relative to Fund (%)"] = np.round((df["Spectral Amplitude"] / fund_amp) * 100.0, 2)

    return df


def generate_spectrum_interpretation(
    waveform: str,
    f0: float,
    peaks_df: pd.DataFrame,
    fs: float
) -> str:
    """
    Produces concise academic commentary explaining the physics and mathematics
    observed in the frequency spectrum.
    """
    wf = waveform.lower()
    base_notes = [
        f"**Fundamental Frequency ($f_0$):** {f0:.2f} Hz. The harmonic peaks appear at exact integer multiples $f_n = n \\cdot f_0$."
    ]

    if "square" in wf:
        base_notes.append(
            "**Odd Harmonic Dominance:** Square waves possess half-wave odd symmetry ($x(t + T/2) = -x(t)$). "
            "Consequently, all even harmonics ($2f_0, 4f_0, \\dots$) have zero energy. "
            "The amplitude of odd harmonics rolls off slowly at $1/n$ ($-20$ dB/decade)."
        )
    elif "sawtooth" in wf:
        base_notes.append(
            "**All Harmonics Present:** The linear ramp has no half-wave symmetry, so both odd and even harmonics "
            "appear ($f_0, 2f_0, 3f_0, 4f_0, \\dots$). Like the square wave, its jump discontinuity causes a slow $1/n$ decay."
        )
    elif "triangle" in wf:
        base_notes.append(
            "**Rapid Harmonic Decay ($1/n^2$):** Because triangular waves are continuous (only their first derivative is discontinuous), "
            "their harmonic amplitudes roll off rapidly as $1/n^2$ ($-40$ dB/decade). "
            "Even harmonics are zero due to odd symmetry."
        )
    elif "sine" in wf:
        base_notes.append(
            "**Pure Spectral Line:** An ideal sine wave is a single frequency component. "
            "All energy is concentrated in the fundamental bin at $f_0$, with negligible harmonic distortion."
        )
    elif "pulse" in wf:
        base_notes.append(
            "**Sinc Envelope Sampling:** The spectrum of a rectangular pulse train follows a sampled $\\text{sinc}(n \\cdot d)$ envelope. "
            "Nulls (zeros) occur at harmonics where $n \\cdot d$ is an integer."
        )

    base_notes.append(
        f"**Sampling & Resolution:** Sampling rate is {fs:.1f} Hz (Nyquist limit = {fs/2.0:.1f} Hz). "
        "Sampling an integer number of complete periods ensures zero spectral leakage under rectangular windowing."
    )

    return "\n\n".join(base_notes)
