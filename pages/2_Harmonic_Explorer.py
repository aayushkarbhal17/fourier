"""
Harmonic Explorer — Inspect Individual Fourier Harmonics
Allows isolation and visual analysis of individual sinusoidal terms,
orthogonal decomposition, harmonic magnitude spectra, and coefficient tables.
"""

import streamlit as st
import numpy as np
import pandas as pd

from src.ui_theme import apply_custom_css, render_header, render_kpi
from src.fourier_series import (
    get_coefficients_by_waveform,
    compute_fourier_series,
    compute_harmonic_table
)
from src.signal_generator import generate_time_array
from src.plotting import (
    plot_harmonic_explorer,
    plot_harmonic_coefficients_stem
)
from src.exports import export_coefficients_csv

st.set_page_config(
    page_title="Harmonic Explorer — FourierLab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Sidebar Controls
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.5rem 0;">
        <span class="eng-badge">Harmonic Controls</span>
        <h3 style="margin: 0.2rem 0; color: #00f0ff;">🔍 Explorer Setup</h3>
    </div>
    """, unsafe_allow_html=True)

    wf_options = ["Square Wave", "Sawtooth Wave", "Triangular Wave", "Sine Wave", "Pulse Train"]
    selected_waveform = st.selectbox("Waveform Type", wf_options, index=0)

    duty_cycle = 0.5
    if selected_waveform == "Pulse Train":
        duty_cycle = st.slider("Duty Cycle (d)", 0.05, 0.95, 0.5, 0.05)

    n_max = st.slider("Max Harmonics Considered (N)", min_value=5, max_value=50, value=25, step=1)
    f0 = st.slider("Fundamental Frequency f₀ (Hz)", min_value=0.2, max_value=10.0, value=1.0, step=0.2)
    amplitude = st.slider("Amplitude (A)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    st.markdown("---")
    selected_n = st.slider("Select Harmonic to Inspect (n)", min_value=1, max_value=n_max, value=3, step=1)

render_header(
    title="Harmonic Explorer — Single Harmonic Isolation",
    subtitle=f"Decomposing the {selected_waveform} to inspect the exact mathematical and physical role of harmonic #{selected_n}.",
    badge_text="Signals & Systems Component Inspection",
    badge_color="purple"
)

# Compute Fourier coefficients
a0, a_coeffs, b_coeffs = get_coefficients_by_waveform(selected_waveform, amplitude, n_max, duty_cycle=duty_cycle)
harmonic_df = compute_harmonic_table(f0, a0, a_coeffs, b_coeffs)

# Generate Time Base
t, fs = generate_time_array(f0=f0, num_periods=2.0, samples_per_period=1000)

# Compute full reconstruction
total_recon, harmonics_matrix = compute_fourier_series(
    t, f0, a0, a_coeffs, b_coeffs, return_harmonics=True
)

# Extract selected harmonic signal
idx = selected_n - 1
isolated_h = harmonics_matrix[idx, :] if harmonics_matrix is not None else np.zeros_like(t)

# Cumulative partial sum up to harmonic n
a_sub = a_coeffs[:selected_n]
b_sub = b_coeffs[:selected_n]
cumulative_recon, _ = compute_fourier_series(t, f0, a0, a_sub, b_sub)

# Selected Harmonic Metadata
row = harmonic_df.iloc[idx]
freq_n = float(row["Frequency (Hz)"])
an_val = float(row["a_n (Cosine)"])
bn_val = float(row["b_n (Sine)"])
cn_val = float(row["Magnitude c_n"])
phase_val = float(row["Phase (deg)"])
pwr_val = float(row["Power Share (%)"])
status_str = str(row["Status"])

# KPI Cards
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi("Harmonic Number", f"n = {selected_n}", f"f_{selected_n} = {freq_n:.2f} Hz", "#a855f7")
with k2:
    render_kpi("Magnitude |c_n|", f"{cn_val:.4f}", f"Cosine: {an_val:.3f}, Sine: {bn_val:.3f}", "#00f0ff")
with k3:
    render_kpi("Phase Angle", f"{phase_val:.1f}°", f"{(phase_val * np.pi / 180):.3f} rad", "#38bdf8")
with k4:
    render_kpi("Energy Contribution", f"{pwr_val:.2f}%", "Fraction of signal power", "#10b981")
with k5:
    status_color = "#10b981" if "Active" in status_str else "#64748b"
    render_kpi("Harmonic Status", status_str, "Symmetry-dependent", status_color)

st.markdown("<br>", unsafe_allow_html=True)

# Isolated Harmonic and Partial Sum Charts
fig_explorer = plot_harmonic_explorer(
    t=t,
    total_reconstructed=total_recon,
    harmonic_signal=isolated_h,
    cumulative_signal=cumulative_recon,
    n=selected_n,
    freq_n=freq_n,
    magnitude=cn_val,
    phase_deg=phase_val
)
st.plotly_chart(fig_explorer, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Stem Plot of All Harmonic Magnitudes
col_stem, col_theory = st.columns([1.6, 1.0])
with col_stem:
    is_zero_arr = (harmonic_df["Status"] == "Zero (Null)").values
    fig_stem = plot_harmonic_coefficients_stem(
        harmonic_numbers=harmonic_df["Harmonic (n)"].values,
        magnitudes=harmonic_df["Magnitude c_n"].values,
        is_zero=is_zero_arr
    )
    st.plotly_chart(fig_stem, use_container_width=True)

with col_theory:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #00f0ff; margin-top: 0;">📐 Harmonic Mathematics</h4>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            <strong>Fundamental Relationship:</strong><br>
            Each harmonic component oscillates at an exact integer multiple of the fundamental frequency:
        </p>
        <div style="background: rgba(15,23,42,0.8); padding: 0.5rem; border-radius: 6px; text-align: center; font-family: monospace; color: #38bdf8; margin: 0.5rem 0;">
            f_n = n · f₀ &nbsp;&nbsp;|&nbsp;&nbsp; ω_n = n · ω₀
        </div>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            <strong>Orthogonality:</strong><br>
            The sinusoidal basis functions are mutually orthogonal over any period T:
        </p>
        <div style="background: rgba(15,23,42,0.8); padding: 0.5rem; border-radius: 6px; text-align: center; font-family: monospace; color: #a855f7; margin: 0.5rem 0;">
            ∫₀ᵀ sin(nω₀t) sin(mω₀t) dt = (T/2) · δ_{nm}
        </div>
        <p style="color: #94a3b8; font-size: 0.82rem;">
            Because cross-terms integrate to zero, each harmonic contributes independently to the signal's total energy without interference.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Full Harmonic Coefficient Table & Export
st.markdown("### 📋 Complete Harmonic Coefficient Table")
st.markdown("Inspect all Fourier coefficients up to harmonic N, highlighting null coefficients canceled by waveform symmetry.")

# Highlight zero rows in dataframe display
st.dataframe(
    harmonic_df,
    use_container_width=True,
    height=320,
    column_config={
        "Power Share (%)": st.column_config.ProgressColumn(
            "Power Share (%)",
            format="%.2f%%",
            min_value=0,
            max_value=100
        )
    }
)

csv_coeffs = export_coefficients_csv(harmonic_df)
st.download_button(
    label="📥 Download Harmonic Coefficients (CSV)",
    data=csv_coeffs,
    file_name=f"fourierlab_{selected_waveform.lower().replace(' ', '_')}_coefficients_N{n_max}.csv",
    mime="text/csv"
)
