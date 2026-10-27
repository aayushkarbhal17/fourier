"""
Signal Generator Module for FourierLab
Generates exact ideal reference signals strictly distinct from Fourier approximations.
"""

from typing import Tuple
import numpy as np


def generate_time_array(
    f0: float,
    num_periods: float = 2.0,
    samples_per_period: int = 1000
) -> Tuple[np.ndarray, float]:
    """
    Generates an evenly spaced time array covering specified number of periods.
    Returns:
        t: 1D NumPy array of time points in seconds
        fs: Sampling frequency in Hz
    """
    f0 = max(0.01, float(f0))
    num_periods = max(0.5, float(num_periods))
    samples_per_period = max(100, int(samples_per_period))

    period = 1.0 / f0
    total_duration = num_periods * period
    total_samples = int(np.round(num_periods * samples_per_period))
    
    t = np.linspace(0.0, total_duration, total_samples, endpoint=False)
    fs = float(samples_per_period * f0)
    return t, fs


def generate_ideal_square(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Ideal square wave with odd symmetry:
    +A during first half-period [0, T/2), -A during second half-period [T/2, T).
    """
    theta = 2.0 * np.pi * f0 * t + phase_offset_rad
    theta_mod = np.mod(theta, 2.0 * np.pi)
    signal = np.where(theta_mod < np.pi, amplitude, -amplitude)
    return signal + dc_offset


def generate_ideal_sawtooth(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Ideal rising sawtooth ramp with odd symmetry:
    Rises from 0 to +A on [0, T/2), and from -A to 0 on [T/2, T).
    """
    theta = 2.0 * np.pi * f0 * t + phase_offset_rad
    theta_mod = np.mod(theta, 2.0 * np.pi)
    
    # Map theta_mod in [0, 2pi) to [-pi, pi) for standard centered odd ramp
    # When theta_mod in [0, pi): ramp is (theta_mod / pi) * amplitude
    # When theta_mod in [pi, 2pi): ramp is ((theta_mod - 2pi) / pi) * amplitude
    signal = np.where(
        theta_mod < np.pi,
        (theta_mod / np.pi) * amplitude,
        ((theta_mod - 2.0 * np.pi) / np.pi) * amplitude
    )
    return signal + dc_offset


def generate_ideal_triangle(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Ideal symmetric continuous triangle wave with odd symmetry:
    Starts at 0 at t=0, rises to +A at T/4, falls to -A at 3T/4, returns to 0 at T.
    """
    theta = 2.0 * np.pi * f0 * t + phase_offset_rad
    theta_mod = np.mod(theta, 2.0 * np.pi)
    
    # Quarter cycles
    q1 = theta_mod < (0.5 * np.pi)
    q23 = (theta_mod >= (0.5 * np.pi)) & (theta_mod < (1.5 * np.pi))
    q4 = theta_mod >= (1.5 * np.pi)

    signal = np.zeros_like(theta_mod)
    # 0 to pi/2: rises from 0 to A
    signal[q1] = (theta_mod[q1] / (0.5 * np.pi)) * amplitude
    # pi/2 to 3pi/2: falls from A to -A
    signal[q23] = amplitude - ((theta_mod[q23] - 0.5 * np.pi) / np.pi) * (2.0 * amplitude)
    # 3pi/2 to 2pi: rises from -A to 0
    signal[q4] = -amplitude + ((theta_mod[q4] - 1.5 * np.pi) / (0.5 * np.pi)) * amplitude

    return signal + dc_offset


def generate_ideal_sine(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Ideal single-tone sinusoid: A * sin(2*pi*f0*t + phase) + dc_offset.
    """
    return amplitude * np.sin(2.0 * np.pi * f0 * t + phase_offset_rad) + dc_offset


def generate_ideal_pulse(
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    duty_cycle: float = 0.5,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0
) -> np.ndarray:
    """
    Ideal rectangular pulse train with adjustable duty cycle d in (0.01, 0.99):
    Signal takes value A on [0, d*T) and 0 on [d*T, T).
    """
    d = float(np.clip(duty_cycle, 0.01, 0.99))
    theta = 2.0 * np.pi * f0 * t + phase_offset_rad
    theta_mod = np.mod(theta, 2.0 * np.pi)
    signal = np.where(theta_mod < (2.0 * np.pi * d), amplitude, 0.0)
    return signal + dc_offset


def generate_ideal_signal(
    waveform: str,
    t: np.ndarray,
    f0: float,
    amplitude: float = 1.0,
    phase_offset_rad: float = 0.0,
    dc_offset: float = 0.0,
    duty_cycle: float = 0.5
) -> np.ndarray:
    """
    Dispatch function to generate the ideal analytical reference waveform.
    """
    wf_clean = waveform.strip().lower()
    if "square" in wf_clean:
        return generate_ideal_square(t, f0, amplitude, phase_offset_rad, dc_offset)
    elif "sawtooth" in wf_clean:
        return generate_ideal_sawtooth(t, f0, amplitude, phase_offset_rad, dc_offset)
    elif "triangle" in wf_clean or "triangular" in wf_clean:
        return generate_ideal_triangle(t, f0, amplitude, phase_offset_rad, dc_offset)
    elif "sine" in wf_clean:
        return generate_ideal_sine(t, f0, amplitude, phase_offset_rad, dc_offset)
    elif "pulse" in wf_clean:
        return generate_ideal_pulse(t, f0, amplitude, duty_cycle, phase_offset_rad, dc_offset)
    else:
        return generate_ideal_square(t, f0, amplitude, phase_offset_rad, dc_offset)
