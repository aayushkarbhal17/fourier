"""
Unit tests for Fourier Series calculations and analytical coefficients.
Verifies numerical accuracy against classical analytical results.
"""

import numpy as np
import pytest
from src.fourier_series import (
    get_square_coefficients,
    get_sawtooth_coefficients,
    get_triangle_coefficients,
    get_sine_coefficients,
    get_pulse_train_coefficients,
    compute_fourier_series,
    compute_harmonic_table,
    compute_gibbs_overshoot
)
from src.signal_generator import generate_time_array, generate_ideal_square


def test_square_coefficients_analytical():
    """
    Square wave with amplitude A=2.0:
    b_n = 4*A / (n*pi) for odd n; 0 for even n. All a_n = 0, a_0 = 0.
    """
    A = 2.0
    N = 10
    a0, an, bn = get_square_coefficients(A, N)

    assert a0 == 0.0
    assert np.all(an == 0.0)
    assert len(bn) == N

    # Check odd harmonics
    expected_b1 = 4.0 * A / (1.0 * np.pi)
    expected_b3 = 4.0 * A / (3.0 * np.pi)
    expected_b5 = 4.0 * A / (5.0 * np.pi)

    assert pytest.approx(bn[0], rel=1e-5) == expected_b1
    assert pytest.approx(bn[2], rel=1e-5) == expected_b3
    assert pytest.approx(bn[4], rel=1e-5) == expected_b5

    # Check even harmonics are strictly zero
    assert bn[1] == 0.0
    assert bn[3] == 0.0
    assert bn[5] == 0.0


def test_sawtooth_coefficients_analytical():
    """
    Rising sawtooth ramp:
    b_n = (2*A / (n*pi)) * (-1)^(n+1).
    All a_n = 0, a_0 = 0.
    """
    A = 1.5
    N = 8
    a0, an, bn = get_sawtooth_coefficients(A, N, rising=True)

    assert a0 == 0.0
    assert np.all(an == 0.0)

    # b_1 should be + 2*A / pi
    assert pytest.approx(bn[0], rel=1e-5) == (2.0 * A / (1.0 * np.pi))
    # b_2 should be - 2*A / (2*pi) = - A / pi
    assert pytest.approx(bn[1], rel=1e-5) == (-2.0 * A / (2.0 * np.pi))
    # b_3 should be + 2*A / (3*pi)
    assert pytest.approx(bn[2], rel=1e-5) == (2.0 * A / (3.0 * np.pi))
    # b_4 should be - 2*A / (4*pi)
    assert pytest.approx(bn[3], rel=1e-5) == (-2.0 * A / (4.0 * np.pi))


def test_triangle_coefficients_analytical():
    """
    Triangular wave:
    b_n = (8*A / (pi^2 * n^2)) * (-1)^((n-1)/2) for odd n; 0 for even n.
    Decays rapidly as 1/n^2.
    """
    A = 3.0
    N = 9
    a0, an, bn = get_triangle_coefficients(A, N)

    assert a0 == 0.0
    assert np.all(an == 0.0)

    expected_b1 = 8.0 * A / (np.pi**2 * 1.0)
    expected_b3 = -8.0 * A / (np.pi**2 * 9.0)
    expected_b5 = 8.0 * A / (np.pi**2 * 25.0)

    assert pytest.approx(bn[0], rel=1e-5) == expected_b1
    assert pytest.approx(bn[2], rel=1e-5) == expected_b3
    assert pytest.approx(bn[4], rel=1e-5) == expected_b5

    # Even harmonics are 0
    assert bn[1] == 0.0
    assert bn[3] == 0.0


def test_sine_coefficients():
    """
    Pure sine wave: only fundamental b_1 = A.
    """
    A = 2.5
    N = 5
    a0, an, bn = get_sine_coefficients(A, N)
    assert a0 == 0.0
    assert np.all(an == 0.0)
    assert pytest.approx(bn[0]) == A
    assert np.all(bn[1:] == 0.0)


def test_pulse_train_coefficients():
    """
    50% duty cycle pulse train: DC component is A/2.
    """
    A = 2.0
    N = 6
    a0, an, bn = get_pulse_train_coefficients(A, N, duty_cycle=0.5)
    # a0/2 = A * 0.5 = 1.0
    assert pytest.approx(a0 / 2.0) == 1.0
    # For 50% duty cycle, an = 0 because sin(2*pi*n*0.5) = sin(n*pi) = 0
    assert np.allclose(an, 0.0, atol=1e-12)


def test_fourier_series_shape_and_finite():
    """
    Verifies that compute_fourier_series outputs finite numbers matching array length.
    """
    t, fs = generate_time_array(f0=2.0, num_periods=2.0, samples_per_period=500)
    a0, an, bn = get_square_coefficients(amplitude=1.0, n_harmonics=25)
    recon, h_matrix = compute_fourier_series(t, f0=2.0, a0=a0, a_coeffs=an, b_coeffs=bn, return_harmonics=True)

    assert recon.shape == t.shape
    assert np.all(np.isfinite(recon))
    assert h_matrix is not None
    assert h_matrix.shape == (25, len(t))
    assert np.all(np.isfinite(h_matrix))


def test_gibbs_overshoot_square():
    """
    Verifies Gibbs overshoot detection near 1.17898 * A for square wave with large N.
    """
    A = 1.0
    t, _ = generate_time_array(f0=1.0, num_periods=1.0, samples_per_period=2000)
    a0, an, bn = get_square_coefficients(amplitude=A, n_harmonics=75)
    recon, _ = compute_fourier_series(t, f0=1.0, a0=a0, a_coeffs=an, b_coeffs=bn)

    gibbs_info = compute_gibbs_overshoot(A, recon)
    # Peak value should approach ~1.17898
    assert gibbs_info["actual_max"] > 1.15
    assert gibbs_info["actual_max"] < 1.25
    assert gibbs_info["overshoot_percent"] > 15.0


def test_harmonic_table_generation():
    """
    Verifies harmonic table structure, zero flags, and columns.
    """
    f0 = 5.0
    a0, an, bn = get_square_coefficients(amplitude=1.0, n_harmonics=10)
    df = compute_harmonic_table(f0, a0, an, bn)

    assert len(df) == 10
    assert "Harmonic (n)" in df.columns
    assert "Frequency (Hz)" in df.columns
    assert "Magnitude c_n" in df.columns
    assert "Status" in df.columns

    # Harmonic 2 should be null/zero
    assert df.iloc[1]["Status"] == "Zero (Null)"
    # Harmonic 1 should be active
    assert df.iloc[0]["Status"] == "Active Harmonic"
    # Frequency for n=3 should be 15 Hz
    assert df.iloc[2]["Frequency (Hz)"] == 15.0
