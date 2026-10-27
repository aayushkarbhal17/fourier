"""
Frequency Analysis and FFT Dashboard
Calculates single-sided amplitude spectra, identifies harmonic peaks,
analyzes spectral leakage, windowing, and harmonic decay rates (1/n vs 1/n^2).
"""

import streamlit as st
import numpy as np
import pandas as pd

from src.ui_theme import apply_custom_css, render_header, render_kpi
from src.fourier_series import get_coefficients_by_waveform, compute_fourier_series
from src.signal_generator import generate_time_array, generate_ideal_signal
from src.frequency_analysis import (
    compute_single_sided_fft,
    find_spectral_peaks,
    generate_spectrum_interpretation
)
from src.plotting import plot_frequency_spectrum
from src.exports import export_spectrum_csv

st.set_page_config(
    page_title="Frequency Analysis — FourierLab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Sidebar Setup
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.5rem 0;">
        <span class="eng-badge">Spectral Controls</span>
        <h3 style="margin: 0.2rem 0; color: #00f0ff;">📊 FFT Settings</h3>
    </div>
    """, unsafe_allow_html=True)

    wf_options = ["Square Wave", "Sawtooth Wave", "Triangular Wave", "Sine Wave", "Pulse Train"]
    selected_waveform = st.selectbox("Waveform Type", wf_options, index=0)

    duty_cycle = 0.5
    if selected_waveform == "Pulse Train":
        duty_cycle = st.slider("Duty Cycle (d)", 0.05, 0.95, 0.5, 0.05)

    n_harmonics = st.slider("Harmonics in Reconstruction (N)", min_value=1, max_value=60, value=15, step=1)
    f0 = st.slider("Fundamental Frequency f₀ (Hz)", min_value=0.5, max_value=15.0, value=2.0, step=0.5)
    amplitude = st.slider("Signal Amplitude (A)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    st.markdown("---")
    st.markdown("### 🪟 FFT Parameters")
    window_choice = st.selectbox(
        "Windowing Function",
        ["Rectangular (None)", "Hann", "Hamming", "Blackman"],
        index=0,
        help="Applies a tapering window to reduce spectral leakage when non-integer periods are sampled."
    )
    window_key = window_choice.split()[0].lower()

    num_periods = st.slider("Acquisition Periods", min_value=1.0, max_value=10.0, value=4.0, step=0.5)
    samples_per_period = st.select_slider("Sampling Points/Period", options=[500, 1000, 2000, 3000], value=1000)

render_header(
    title="Frequency Analysis & Fast Fourier Transform",
    subtitle=f"Comparing discrete single-sided frequency spectra of the ideal continuous-reference vs N={n_harmonics} Fourier reconstruction.",
    badge_text="Frequency-Domain Spectral Analysis",
    badge_color="green"
)

# Generate Time Base and Waveforms
t, fs = generate_time_array(f0=f0, num_periods=num_periods, samples_per_period=samples_per_period)
ideal_sig = generate_ideal_signal(selected_waveform, t, f0, amplitude, duty_cycle=duty_cycle)
a0, an, bn = get_coefficients_by_waveform(selected_waveform, amplitude, n_harmonics, duty_cycle=duty_cycle)
recon_sig, _ = compute_fourier_series(t, f0, a0, an, bn)

# Compute Single-Sided FFT
freqs_ideal, amps_ideal, meta_ideal = compute_single_sided_fft(ideal_sig, fs, window_type=window_key)
freqs_recon, amps_recon, meta_recon = compute_single_sided_fft(recon_sig, fs, window_type=window_key)

# Identify Spectral Peaks
peaks_df = find_spectral_peaks(freqs_recon, amps_recon, f0=f0, amplitude_threshold=0.015, max_peaks=12)

# KPI Summary
f_nyquist = fs / 2.0
f_resolution = fs / len(t)
max_display_f = min(f_nyquist, max(30.0, float(n_harmonics * f0 * 1.5)))

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi("Sampling Rate f_s", f"{fs:.1f} Hz", "Discrete time rate", "#00f0ff")
with k2:
    render_kpi("Nyquist Limit", f"{f_nyquist:.1f} Hz", "f_s / 2 limit", "#a855f7")
with k3:
    render_kpi("Frequency Resolution", f"{f_resolution:.3f} Hz", "Bin spacing Δf = f_s / N", "#10b981")
with k4:
    render_kpi("Harmonics In Recon", f"N = {n_harmonics}", f"Highest: {n_harmonics * f0:.1f} Hz", "#38bdf8")
with k5:
    render_kpi("Detected Peaks", f"{len(peaks_df)} Peaks", f"Threshold: 0.015 V", "#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Frequency Spectrum Plot
fig_spec = plot_frequency_spectrum(
    freqs_ideal=freqs_ideal,
    amps_ideal=amps_ideal,
    freqs_recon=freqs_recon,
    amps_recon=amps_recon,
    f0=f0,
    max_freq_display=max_display_f,
    peaks_df=peaks_df
)
st.plotly_chart(fig_spec, use_container_width=True)

# Two-Column Section: Harmonic Peaks Table and Academic Interpretation
c_table, c_notes = st.columns([1.3, 1.2])

with c_table:
    st.markdown("### 🎯 Detected Harmonic Peaks")
    st.markdown("Harmonic peaks isolated from the FFT of the reconstructed signal:")
    if not peaks_df.empty:
        st.dataframe(
            peaks_df,
            use_container_width=True,
            height=300,
            column_config={
                "Relative to Fund (%)": st.column_config.ProgressColumn(
                    "Relative to Fund (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            }
        )
    else:
        st.info("No harmonic peaks exceeded the detection threshold.")

    # Export spectrum CSV
    csv_spec = export_spectrum_csv(freqs_recon, amps_ideal, amps_recon, max_freq=max_display_f)
    st.download_button(
        label="📥 Download Frequency Spectrum (CSV)",
        data=csv_spec,
        file_name=f"fourierlab_{selected_waveform.lower().replace(' ', '_')}_spectrum.csv",
        mime="text/csv"
    )

with c_notes:
    st.markdown("### 🧠 Physical & Spectral Interpretation")
    interp_text = generate_spectrum_interpretation(selected_waveform, f0, peaks_df, fs)
    st.markdown(f"""
    <div class="eng-card">
        {interp_text}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Deep-Dive Educational Callout: Harmonic Decay Comparison
st.markdown("### 📉 Mathematical Decay Rates: 1/n vs 1/n²")
col_d1, col_d2 = st.columns(2)
with col_d1:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #00f0ff; margin-top: 0;">Discontinuous Signals (Square / Sawtooth): 1/n Decay</h4>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            A square wave or sawtooth wave contains instantaneous jump discontinuities in the time domain:
        </p>
        <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
            <li>Harmonic coefficients decrease slowly: <code>|bₙ| ∝ 1/n</code></li>
            <li>Roll-off slope: <strong>-20 dB / decade</strong> (or -6 dB / octave).</li>
            <li>Consequence: High-frequency harmonics carry substantial energy. Truncating the series causes Gibbs ringing.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #10b981; margin-top: 0;">Continuous Piecewise Signals (Triangular): 1/n² Decay</h4>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6;">
            A triangular wave is continuous everywhere; only its derivative has step jumps:
        </p>
        <ul style="color: #94a3b8; font-size: 0.85rem; line-height: 1.6;">
            <li>Harmonic coefficients drop rapidly: <code>|bₙ| ∝ 1/n²</code></li>
            <li>Roll-off slope: <strong>-40 dB / decade</strong> (or -12 dB / octave).</li>
            <li>Consequence: The fundamental carries > 99% of total signal energy. Harmonics vanish extremely quickly.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
