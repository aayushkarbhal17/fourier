"""
Unit tests for quantitative metrics, error analysis, and export formats.
"""

import numpy as np
import pytest
from src.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_max_error,
    calculate_signal_to_error_ratio,
    calculate_parseval_power_share,
    compute_convergence_data
)
from src.exports import export_waveforms_csv, export_spectrum_csv, generate_summary_report
from src.fourier_series import get_square_coefficients


def test_error_metrics_identical_signals():
    """
    When reconstructed signal perfectly equals ideal signal:
    RMSE = 0, MAE = 0, Max Error = 0, SER is arbitrarily large.
    """
    arr = np.array([1.0, -1.0, 2.0, -2.0, 0.5])
    assert calculate_rmse(arr, arr) == 0.0
    assert calculate_mae(arr, arr) == 0.0
    assert calculate_max_error(arr, arr) == 0.0
    assert calculate_signal_to_error_ratio(arr, arr) >= 100.0


def test_error_metrics_known_difference():
    """
    Test with known numerical deviation of constant delta = 0.5.
    """
    ideal = np.array([1.0, 2.0, 3.0, 4.0])
    recon = ideal + 0.5

    assert pytest.approx(calculate_rmse(ideal, recon)) == 0.5
    assert pytest.approx(calculate_mae(ideal, recon)) == 0.5
    assert pytest.approx(calculate_max_error(ideal, recon)) == 0.5


def test_parseval_power_share_convergence():
    """
    For a square wave of amplitude 1.0, theoretical power is 1.0.
    With N=1: power share should be (4/pi)^2 * 0.5 = 8 / pi^2 ~= 81.06%.
    With N=3: adds 1/9 * 81.06% = ~90.06%.
    With N=50: approaches ~99%.
    """
    a0, an1, bn1 = get_square_coefficients(amplitude=1.0, n_harmonics=1)
    share1 = calculate_parseval_power_share("square", 1.0, a0, an1, bn1)
    assert pytest.approx(share1, rel=0.01) == (8.0 / (np.pi ** 2) * 100.0)

    a0, an50, bn50 = get_square_coefficients(amplitude=1.0, n_harmonics=50)
    share50 = calculate_parseval_power_share("square", 1.0, a0, an50, bn50)
    assert share50 > share1
    assert share50 > 98.0


def test_compute_convergence_data():
    """
    Verifies that compute_convergence_data returns a valid DataFrame
    with decreasing RMSE as N increases.
    """
    df = compute_convergence_data(waveform="square", amplitude=1.0, f0=1.0, max_harmonics=15)
    assert not df.empty
    assert "Harmonics (N)" in df.columns
    assert "RMSE" in df.columns
    assert "MAE" in df.columns

    # Verify that RMSE decreases monotonically from first to last point
    rmse_first = df.iloc[0]["RMSE"]
    rmse_last = df.iloc[-1]["RMSE"]
    assert rmse_last < rmse_first


def test_exports_csv_and_report():
    """
    Verifies CSV export strings and summary report generation.
    """
    t = np.array([0.0, 0.1, 0.2])
    ideal = np.array([1.0, 1.0, -1.0])
    recon = np.array([0.9, 1.1, -0.95])

    csv_data = export_waveforms_csv(t, ideal, recon)
    assert "time_seconds,ideal_reference,fourier_reconstructed,approximation_error" in csv_data
    assert "0.0,1.0,0.9,0.1" in csv_data

    # Report generation
    params = {"waveform": "Square Wave", "n_harmonics": 15, "f0": 1.0, "amplitude": 1.0}
    metrics = {"rmse": 0.12, "mae": 0.08, "max_error": 0.18, "ser_db": 18.5, "energy_share_pct": 98.5}
    student = {"name": "Test Student", "roll": "12345"}
    report = generate_summary_report(params, metrics, student)

    assert "FOURIERLAB: INTERACTIVE FOURIER SERIES SIMULATION REPORT" in report
    assert "Test Student" in report
    assert "18.50 dB" in report
