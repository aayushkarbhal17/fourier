"""
Quantitative Metrics and Error Analysis Module for FourierLab
Computes RMSE, MAE, Max Error, Signal-to-Error Ratio (SER), Parseval energy capture,
and convergence curves across harmonic counts.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from .fourier_series import get_coefficients_by_waveform, compute_fourier_series
from .signal_generator import generate_time_array, generate_ideal_signal


def calculate_rmse(ideal: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes Root Mean Square Error (RMSE) between ideal reference and reconstructed series:
    RMSE = sqrt( (1/M) * sum( (x_ideal - x_recon)^2 ) )
    """
    if len(ideal) == 0 or len(reconstructed) == 0:
        return 0.0
    return float(np.sqrt(np.mean((ideal - reconstructed) ** 2)))


def calculate_mae(ideal: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes Mean Absolute Error (MAE):
    MAE = (1/M) * sum( |x_ideal - x_recon| )
    """
    if len(ideal) == 0 or len(reconstructed) == 0:
        return 0.0
    return float(np.mean(np.abs(ideal - reconstructed)))


def calculate_max_error(ideal: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes Maximum Absolute Error:
    MaxError = max( |x_ideal - x_recon| )
    Highlights Gibbs peak overshoot near jump discontinuities.
    """
    if len(ideal) == 0 or len(reconstructed) == 0:
        return 0.0
    return float(np.max(np.abs(ideal - reconstructed)))


def calculate_signal_to_error_ratio(ideal: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Computes Signal-to-Reconstruction-Error Ratio (SER) in decibels:
    SER (dB) = 10 * log10( sum(x_ideal^2) / sum((x_ideal - x_recon)^2) )
    
    Note: This measures deterministic approximation fidelity, distinct from
    stochastic noise SNR in communications channels.
    """
    ideal_energy = float(np.sum(ideal ** 2))
    error_energy = float(np.sum((ideal - reconstructed) ** 2))

    if error_energy < 1e-12:
        return 100.0  # Essentially infinite fidelity (e.g. pure sine wave with N>=1)
    if ideal_energy < 1e-12:
        return 0.0

    ser = 10.0 * np.log10(ideal_energy / error_energy)
    return float(ser)


def get_ideal_signal_power(waveform: str, amplitude: float) -> float:
    """
    Returns theoretical average power (mean-square value) P = (1/T) integral x^2(t) dt:
    - Square: A^2
    - Sawtooth: A^2 / 3
    - Triangle: A^2 / 3
    - Sine: A^2 / 2
    """
    wf = waveform.lower()
    a_sq = amplitude ** 2
    if "square" in wf:
        return a_sq
    elif "sawtooth" in wf or "triangle" in wf:
        return a_sq / 3.0
    elif "sine" in wf:
        return a_sq / 2.0
    else:
        return a_sq


def calculate_parseval_power_share(
    waveform: str,
    amplitude: float,
    a0: float,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray
) -> float:
    """
    Calculates percentage of total theoretical signal energy captured by N harmonics
    via Parseval's Theorem:
    P_reconstructed = (a0/2)^2 + (1/2) * sum_{n=1}^N (a_n^2 + b_n^2)
    Share (%) = (P_reconstructed / P_ideal) * 100
    """
    ideal_power = get_ideal_signal_power(waveform, amplitude)
    if ideal_power < 1e-12:
        return 100.0

    ac_power = 0.5 * np.sum(a_coeffs ** 2 + b_coeffs ** 2)
    dc_power = (a0 / 2.0) ** 2
    total_recon_power = dc_power + ac_power

    share = (total_recon_power / ideal_power) * 100.0
    return float(np.clip(share, 0.0, 100.0))


def compute_all_metrics(
    waveform: str,
    amplitude: float,
    ideal: np.ndarray,
    reconstructed: np.ndarray,
    a0: float,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray
) -> Dict[str, Any]:
    """
    Computes a complete suite of quantitative fidelity and energy metrics.
    """
    rmse = calculate_rmse(ideal, reconstructed)
    mae = calculate_mae(ideal, reconstructed)
    max_err = calculate_max_error(ideal, reconstructed)
    ser_db = calculate_signal_to_error_ratio(ideal, reconstructed)
    energy_captured_pct = calculate_parseval_power_share(waveform, amplitude, a0, a_coeffs, b_coeffs)

    return {
        "rmse": rmse,
        "mae": mae,
        "max_error": max_err,
        "ser_db": ser_db,
        "energy_share_pct": energy_captured_pct,
        "harmonic_count": len(a_coeffs)
    }


def compute_convergence_data(
    waveform: str,
    amplitude: float = 1.0,
    f0: float = 1.0,
    max_harmonics: int = 40,
    duty_cycle: float = 0.5
) -> pd.DataFrame:
    """
    Evaluates reconstruction error trends across increasing harmonic counts N = 1, 2, ..., max_harmonics.
    Used for comparative convergence charts and student laboratory analysis.
    """
    t, _ = generate_time_array(f0, num_periods=1.0, samples_per_period=1200)
    ideal = generate_ideal_signal(waveform, t, f0, amplitude, duty_cycle=duty_cycle)

    rows = []
    # Sample harmonics logarithmically or linearly
    step = 1 if max_harmonics <= 30 else (2 if max_harmonics <= 60 else 3)
    n_values = list(range(1, max_harmonics + 1, step))
    if max_harmonics not in n_values:
        n_values.append(max_harmonics)

    for n in n_values:
        a0, an, bn = get_coefficients_by_waveform(waveform, amplitude, n, duty_cycle=duty_cycle)
        recon, _ = compute_fourier_series(t, f0, a0, an, bn)
        
        rmse = calculate_rmse(ideal, recon)
        mae = calculate_mae(ideal, recon)
        max_err = calculate_max_error(ideal, recon)
        ser = calculate_signal_to_error_ratio(ideal, recon)

        rows.append({
            "Harmonics (N)": n,
            "RMSE": round(rmse, 5),
            "MAE": round(mae, 5),
            "Max Absolute Error": round(max_err, 5),
            "SER (dB)": round(ser, 2)
        })

    return pd.DataFrame(rows)
