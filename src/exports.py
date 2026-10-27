"""
Data Export and Reporting Module for FourierLab
Provides clean CSV serialization and comprehensive academic simulation reports.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd


def export_waveforms_csv(
    t: np.ndarray,
    ideal: np.ndarray,
    reconstructed: np.ndarray
) -> str:
    """
    Serializes time-domain signals to CSV format.
    """
    df = pd.DataFrame({
        "time_seconds": np.round(t, 6),
        "ideal_reference": np.round(ideal, 6),
        "fourier_reconstructed": np.round(reconstructed, 6),
        "approximation_error": np.round(ideal - reconstructed, 6)
    })
    return df.to_csv(index=False)


def export_coefficients_csv(df_coefficients: pd.DataFrame) -> str:
    """
    Serializes harmonic coefficients table to CSV.
    """
    return df_coefficients.to_csv(index=False)


def export_spectrum_csv(
    freqs: np.ndarray,
    ideal_amps: np.ndarray,
    recon_amps: np.ndarray,
    max_freq: Optional[float] = None
) -> str:
    """
    Serializes frequency spectrum data to CSV up to max_freq.
    """
    if max_freq is not None:
        mask = freqs <= max_freq
        f_sub = freqs[mask]
        i_sub = ideal_amps[mask]
        r_sub = recon_amps[mask]
    else:
        f_sub = freqs
        i_sub = ideal_amps
        r_sub = recon_amps

    df = pd.DataFrame({
        "frequency_hz": np.round(f_sub, 3),
        "ideal_amplitude": np.round(i_sub, 6),
        "reconstructed_amplitude": np.round(r_sub, 6)
    })
    return df.to_csv(index=False)


def generate_summary_report(
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    student_info: Optional[Dict[str, str]] = None
) -> str:
    """
    Builds a professional, comprehensive academic simulation report
    suitable for mini-project submission and viva demonstration.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    student_section = ""
    if student_info:
        student_section = f"""
================================================================================
ACADEMIC SUBMISSION DETAILS
================================================================================
Student Name       : {student_info.get('name', 'N/A')}
Roll / Reg Number  : {student_info.get('roll', 'N/A')}
Department         : {student_info.get('department', 'Signals and Systems / ECE / EE')}
Institution        : {student_info.get('college', 'Engineering College')}
Academic Year      : {student_info.get('year', '2025-2026')}
"""

    report = f"""================================================================================
FOURIERLAB: INTERACTIVE FOURIER SERIES SIMULATION REPORT
Signals and Systems Mini Project
Generated on: {now_str}
================================================================================{student_section}
================================================================================
1. SIMULATION PARAMETERS
================================================================================
Waveform Type          : {params.get('waveform', 'Square Wave')}
Harmonics Included (N) : {params.get('n_harmonics', 15)}
Fundamental Freq (f0)  : {params.get('f0', 1.0):.3f} Hz
Fundamental Period (T) : {1.0 / max(1e-6, params.get('f0', 1.0)):.4f} seconds
Angular Freq (omega0)  : {2.0 * np.pi * params.get('f0', 1.0):.4f} rad/s
Peak Amplitude (A)     : {params.get('amplitude', 1.0):.2f}
Phase Offset           : {params.get('phase_deg', 0.0):.1f} degrees ({params.get('phase_rad', 0.0):.3f} rad)
DC Bias Offset         : {params.get('dc_offset', 0.0):.2f}
Periods Rendered       : {params.get('num_periods', 2.0):.1f}
Sampling Rate (fs)     : {params.get('fs', 1000.0):.1f} Hz
Total Discrete Samples : {params.get('total_samples', 2000)}

================================================================================
2. QUANTITATIVE ACCURACY & FIDELITY METRICS
================================================================================
Root Mean Square Error (RMSE)       : {metrics.get('rmse', 0.0):.6f}
Mean Absolute Error (MAE)           : {metrics.get('mae', 0.0):.6f}
Maximum Absolute Error (Peak Ripple): {metrics.get('max_error', 0.0):.6f}
Signal-to-Error Ratio (SER)         : {metrics.get('ser_db', 0.0):.2f} dB
Parseval Energy Captured            : {metrics.get('energy_share_pct', 0.0):.2f}%

================================================================================
3. ACADEMIC & PHYSICAL INTERPRETATION
================================================================================
- Mathematical Basis:
  The signal is approximated by trigonometric Fourier partial sum:
  x_N(t) = a0/2 + sum_{{n=1}}^{{N}} [ a_n * cos(n*w0*t) + b_n * sin(n*w0*t) ]

- Convergence Behavior:
  As harmonic count N increases, the mean-square error (RMSE) monotonically
  approaches zero in accordance with Parseval's theorem.

- Discontinuity & Gibbs Phenomenon:
  For discontinuous waveforms (such as square and sawtooth waves), jump
  discontinuities exhibit the Gibbs phenomenon: an asymptotic overshoot
  of approximately 8.95% of the total jump height that does not vanish
  as N -> infinity, but rather compresses in width adjacent to the jump.

- Symmetry & Harmonics:
  Signals with odd half-wave symmetry cancel all even harmonics (n = 2, 4, 6...),
  concentrating all non-DC spectral energy exclusively in odd harmonic multiples.

================================================================================
End of FourierLab Simulation Report
================================================================================
"""
    return report.strip()
