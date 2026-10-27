"""
Fourier Series Mathematical Module for FourierLab
Provides exact analytical coefficient computation, trigonometric series synthesis,
individual harmonic extraction, and Gibbs phenomenon analysis.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def get_square_coefficients(amplitude: float, n_harmonics: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes trigonometric Fourier coefficients for an ideal square wave:
    x(t) has odd symmetry, zero mean, range [-A, +A].
    Only odd harmonics are nonzero: b_n = 4*A / (n*pi) for odd n; all a_n = 0, a_0 = 0.
    """
    n_harmonics = max(1, int(n_harmonics))
    a0 = 0.0
    a_coeffs = np.zeros(n_harmonics, dtype=float)
    b_coeffs = np.zeros(n_harmonics, dtype=float)

    n_indices = np.arange(1, n_harmonics + 1)
    is_odd = (n_indices % 2 != 0)
    b_coeffs[is_odd] = (4.0 * amplitude) / (np.pi * n_indices[is_odd])

    return a0, a_coeffs, b_coeffs


def get_sawtooth_coefficients(amplitude: float, n_harmonics: int, rising: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes trigonometric Fourier coefficients for an ideal sawtooth ramp:
    Zero mean, range [-A, +A].
    Odd symmetry: b_n = (2*A / (n*pi)) * (-1)^(n+1) for rising ramp.
    Every integer harmonic is present with 1/n decay.
    """
    n_harmonics = max(1, int(n_harmonics))
    a0 = 0.0
    a_coeffs = np.zeros(n_harmonics, dtype=float)
    
    n_indices = np.arange(1, n_harmonics + 1)
    sign_multiplier = 1.0 if rising else -1.0
    b_coeffs = sign_multiplier * (2.0 * amplitude / (np.pi * n_indices)) * ((-1.0) ** (n_indices + 1))

    return a0, a_coeffs, b_coeffs


def get_triangle_coefficients(amplitude: float, n_harmonics: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes trigonometric Fourier coefficients for an ideal triangular wave:
    Odd symmetry, peak amplitude A, zero mean.
    Only odd harmonics are nonzero with fast 1/n^2 decay:
    b_n = (8*A / (pi^2 * n^2)) * (-1)^((n-1)/2) for odd n; even n are 0.
    """
    n_harmonics = max(1, int(n_harmonics))
    a0 = 0.0
    a_coeffs = np.zeros(n_harmonics, dtype=float)
    b_coeffs = np.zeros(n_harmonics, dtype=float)

    n_indices = np.arange(1, n_harmonics + 1)
    is_odd = (n_indices % 2 != 0)
    
    k = (n_indices[is_odd] - 1) // 2
    b_coeffs[is_odd] = (8.0 * amplitude / (np.pi**2 * (n_indices[is_odd] ** 2))) * ((-1.0) ** k)

    return a0, a_coeffs, b_coeffs


def get_sine_coefficients(amplitude: float, n_harmonics: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes Fourier coefficients for a pure fundamental sine wave:
    Only b_1 = amplitude; all other coefficients are zero.
    """
    n_harmonics = max(1, int(n_harmonics))
    a0 = 0.0
    a_coeffs = np.zeros(n_harmonics, dtype=float)
    b_coeffs = np.zeros(n_harmonics, dtype=float)
    b_coeffs[0] = float(amplitude)

    return a0, a_coeffs, b_coeffs


def get_pulse_train_coefficients(amplitude: float, n_harmonics: int, duty_cycle: float = 0.5) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes trigonometric Fourier coefficients for a periodic rectangular pulse train:
    Amplitude 0 to A, with adjustable duty cycle d in (0.01, 0.99).
    a_0/2 = A * d (DC offset).
    """
    n_harmonics = max(1, int(n_harmonics))
    d = float(np.clip(duty_cycle, 0.01, 0.99))
    a0 = 2.0 * amplitude * d

    n_indices = np.arange(1, n_harmonics + 1)
    two_pi_n_d = 2.0 * np.pi * n_indices * d

    a_coeffs = (amplitude / (np.pi * n_indices)) * np.sin(two_pi_n_d)
    b_coeffs = (amplitude / (np.pi * n_indices)) * (1.0 - np.cos(two_pi_n_d))

    return a0, a_coeffs, b_coeffs


def get_coefficients_by_waveform(
    waveform: str,
    amplitude: float,
    n_harmonics: int,
    duty_cycle: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Dispatch helper to fetch Fourier coefficients for any supported waveform.
    """
    wf_clean = waveform.strip().lower()
    if "square" in wf_clean:
        return get_square_coefficients(amplitude, n_harmonics)
    elif "sawtooth" in wf_clean:
        return get_sawtooth_coefficients(amplitude, n_harmonics, rising=True)
    elif "triangle" in wf_clean or "triangular" in wf_clean:
        return get_triangle_coefficients(amplitude, n_harmonics)
    elif "sine" in wf_clean:
        return get_sine_coefficients(amplitude, n_harmonics)
    elif "pulse" in wf_clean:
        return get_pulse_train_coefficients(amplitude, n_harmonics, duty_cycle=duty_cycle)
    else:
        # Default fallback to square wave
        return get_square_coefficients(amplitude, n_harmonics)


def compute_fourier_series(
    t: np.ndarray,
    f0: float,
    a0: float,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0,
    return_harmonics: bool = False
) -> Tuple[np.ndarray, np.ndarray | None]:
    """
    Reconstructs the partial sum x_N(t) from Fourier coefficients:
    x_N(t) = a0/2 + dc_offset + sum_{n=1}^N [ a_n cos(n*w0*t + n*phi) + b_n sin(n*w0*t + n*phi) ]
    
    If return_harmonics is True, also returns an (N, len(t)) array containing each individual harmonic.
    """
    omega0 = 2.0 * np.pi * f0
    n_harmonics = len(a_coeffs)
    num_samples = len(t)

    # Base DC value
    reconstructed = np.full(num_samples, (a0 / 2.0) + dc_offset, dtype=float)

    harmonics_matrix = np.zeros((n_harmonics, num_samples), dtype=float) if return_harmonics else None

    # Vectorized / efficient calculation across harmonics
    for i in range(n_harmonics):
        n = i + 1
        an = a_coeffs[i]
        bn = b_coeffs[i]
        
        # Skip negligible coefficients for speed
        if abs(an) < 1e-12 and abs(bn) < 1e-12:
            continue

        arg = n * (omega0 * t + phase_offset_rad)
        h_n = an * np.cos(arg) + bn * np.sin(arg)
        
        reconstructed += h_n
        if harmonics_matrix is not None:
            harmonics_matrix[i, :] = h_n

    return reconstructed, harmonics_matrix


def compute_harmonic_table(
    f0: float,
    a0: float,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray
) -> pd.DataFrame:
    """
    Builds an educational table of harmonic parameters:
    Columns: Harmonic (n), Frequency (Hz), Cosine Coeff (a_n), Sine Coeff (b_n),
             Magnitude (c_n), Phase (deg), Zero-flag, Power Fraction (%)
    """
    n_harmonics = len(a_coeffs)
    n_indices = np.arange(1, n_harmonics + 1)
    frequencies = n_indices * f0
    magnitudes = np.sqrt(a_coeffs**2 + b_coeffs**2)
    phases_rad = np.arctan2(a_coeffs, b_coeffs)
    phases_deg = np.degrees(phases_rad)

    # Zero flag: if both |a_n| and |b_n| are negligible (< 1e-9)
    is_zero = (np.abs(a_coeffs) < 1e-9) & (np.abs(b_coeffs) < 1e-9)

    # Power/energy distribution: sum of c_n^2 / 2 for AC + (a0/2)^2
    powers = 0.5 * (magnitudes ** 2)
    total_power = np.sum(powers) + ((a0 / 2.0) ** 2)
    power_pct = (powers / total_power * 100.0) if total_power > 1e-12 else np.zeros_like(powers)

    df = pd.DataFrame({
        "Harmonic (n)": n_indices,
        "Frequency (Hz)": np.round(frequencies, 3),
        "a_n (Cosine)": np.round(a_coeffs, 5),
        "b_n (Sine)": np.round(b_coeffs, 5),
        "Magnitude c_n": np.round(magnitudes, 5),
        "Phase (deg)": np.where(is_zero, 0.0, np.round(phases_deg, 2)),
        "Status": np.where(is_zero, "Zero (Null)", "Active Harmonic"),
        "Power Share (%)": np.round(power_pct, 2)
    })
    return df


def compute_gibbs_overshoot(amplitude: float, reconstructed: np.ndarray) -> Dict[str, float]:
    """
    Analyzes the Gibbs phenomenon overshoot near discontinuities for a square wave.
    Theoretical peak value: ~1.17898 * A (~8.95% of peak-to-peak step 2A above +A).
    """
    max_recon = float(np.max(reconstructed))
    overshoot_abs = max(0.0, max_recon - amplitude)
    overshoot_percent = (overshoot_abs / amplitude) * 100.0 if amplitude > 0 else 0.0
    theoretical_max = amplitude * 1.17898
    theoretical_overshoot_pct = 17.898  # above A (relative to amplitude)

    return {
        "actual_max": max_recon,
        "ideal_amplitude": amplitude,
        "overshoot_abs": overshoot_abs,
        "overshoot_percent": overshoot_percent,
        "theoretical_max": theoretical_max,
        "theoretical_overshoot_pct": theoretical_overshoot_pct,
        "wilbraham_gibbs_constant": 1.08949  # relative to half-step
    }
