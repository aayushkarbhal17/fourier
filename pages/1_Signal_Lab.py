"""
Signal Lab — Interactive Fourier Series Simulator
Primary laboratory environment for parameter tuning, time-domain reconstruction,
Gibbs phenomenon analysis, convergence tracking, and data export.
"""

import streamlit as st
import numpy as np

from src.ui_theme import apply_custom_css, render_header, render_kpi
from src.fourier_series import (
    get_coefficients_by_waveform,
    compute_fourier_series,
    compute_gibbs_overshoot
)
from src.signal_generator import generate_time_array, generate_ideal_signal
from src.metrics import (
    calculate_rmse,
    calculate_mae,
    calculate_max_error,
    calculate_signal_to_error_ratio,
    calculate_parseval_power_share,
    compute_convergence_data
)
from src.plotting import (
    plot_time_domain,
    plot_gibbs_zoom,
    plot_error_convergence,
    create_harmonic_building_animation
)
from src.exports import (
    export_waveforms_csv,
    generate_summary_report
)

st.set_page_config(
    page_title="Signal Lab — FourierLab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

# Default session state initialization
if "sl_waveform" not in st.session_state:
    st.session_state.sl_waveform = "Square Wave"
if "sl_harmonics" not in st.session_state:
    st.session_state.sl_harmonics = 15
if "sl_f0" not in st.session_state:
    st.session_state.sl_f0 = 1.0
if "sl_amplitude" not in st.session_state:
    st.session_state.sl_amplitude = 1.0
if "sl_phase_deg" not in st.session_state:
    st.session_state.sl_phase_deg = 0.0
if "sl_dc_offset" not in st.session_state:
    st.session_state.sl_dc_offset = 0.0
if "sl_periods" not in st.session_state:
    st.session_state.sl_periods = 2.0
if "sl_resolution" not in st.session_state:
    st.session_state.sl_resolution = 1000
if "sl_duty" not in st.session_state:
    st.session_state.sl_duty = 0.5

def reset_defaults():
    st.session_state.sl_waveform = "Square Wave"
    st.session_state.sl_harmonics = 15
    st.session_state.sl_f0 = 1.0
    st.session_state.sl_amplitude = 1.0
    st.session_state.sl_phase_deg = 0.0
    st.session_state.sl_dc_offset = 0.0
    st.session_state.sl_periods = 2.0
    st.session_state.sl_resolution = 1000
    st.session_state.sl_duty = 0.5

# Sidebar Controls
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.25rem 0 0.5rem 0;">
        <span class="eng-badge">Laboratory Controls</span>
        <h3 style="margin: 0.2rem 0; color: #00f0ff;">⚙️ Signal Parameters</h3>
    </div>
    """, unsafe_allow_html=True)

    wf_options = ["Square Wave", "Sawtooth Wave", "Triangular Wave", "Sine Wave", "Pulse Train"]
    current_wf_idx = wf_options.index(st.session_state.sl_waveform) if st.session_state.sl_waveform in wf_options else 0
    selected_waveform = st.selectbox(
        "Periodic Waveform",
        wf_options,
        index=current_wf_idx,
        help="Select standard analytical waveform. Symmetric waveforms produce specific harmonic cancellations."
    )
    st.session_state.sl_waveform = selected_waveform

    duty_cycle = 0.5
    if selected_waveform == "Pulse Train":
        duty_cycle = st.slider(
            "Duty Cycle (d)",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.sl_duty),
            step=0.05,
            help="Fraction of each period where the rectangular pulse is high."
        )
        st.session_state.sl_duty = duty_cycle

    n_harmonics = st.slider(
        "Harmonics Count (N)",
        min_value=1,
        max_value=100,
        value=int(st.session_state.sl_harmonics),
        step=1,
        help="Number of Fourier series terms included in the partial sum x_N(t)."
    )
    st.session_state.sl_harmonics = n_harmonics

    f0 = st.slider(
        "Fundamental Frequency f₀ (Hz)",
        min_value=0.1,
        max_value=20.0,
        value=float(st.session_state.sl_f0),
        step=0.1,
        help="Repetition rate in cycles per second. Period T = 1/f₀."
    )
    st.session_state.sl_f0 = f0

    amplitude = st.slider(
        "Signal Amplitude (A)",
        min_value=0.2,
        max_value=5.0,
        value=float(st.session_state.sl_amplitude),
        step=0.1,
        help="Peak amplitude of the reference periodic waveform."
    )
    st.session_state.sl_amplitude = amplitude

    with st.expander("🔧 Advanced Signal Adjustments", expanded=False):
        phase_deg = st.slider(
            "Phase Offset (degrees)",
            min_value=-180.0,
            max_value=180.0,
            value=float(st.session_state.sl_phase_deg),
            step=5.0,
            help="Time-shift applied to the signal: x(t - t₀)."
        )
        st.session_state.sl_phase_deg = phase_deg

        dc_offset = st.slider(
            "DC Bias Offset",
            min_value=-3.0,
            max_value=3.0,
            value=float(st.session_state.sl_dc_offset),
            step=0.2,
            help="Constant vertical offset added to the total waveform."
        )
        st.session_state.sl_dc_offset = dc_offset

        num_periods = st.slider(
            "Periods to Display",
            min_value=1.0,
            max_value=6.0,
            value=float(st.session_state.sl_periods),
            step=0.5,
            help="Number of complete waveform cycles rendered on time axis."
        )
        st.session_state.sl_periods = num_periods

        res_options = [400, 800, 1000, 1500, 2000, 3000]
        current_res = int(st.session_state.sl_resolution)
        if current_res not in res_options:
            current_res = 1000
        samples_per_period = st.select_slider(
            "Sampling Resolution (pts/period)",
            options=res_options,
            value=current_res,
            help="Number of discrete numerical points calculated per cycle."
        )
        st.session_state.sl_resolution = samples_per_period

        show_harmonics = st.checkbox("Overlay Individual Harmonics", value=False)
        max_overlays = 4
        if show_harmonics:
            max_overlays = st.slider("Harmonics to Overlay", min_value=1, max_value=6, value=3)

    st.markdown("---")
    st.button("↺ Reset to Defaults", on_click=reset_defaults, use_container_width=True)

# Main Page Layout
render_header(
    title="Signal Lab — Fourier Series Simulator",
    subtitle=f"Interactive synthesis of a {selected_waveform} using N={n_harmonics} Fourier harmonics at f₀ = {f0:.2f} Hz.",
    badge_text="Interactive Laboratory Environment",
    badge_color="cyan"
)

# Convert phase to radians
phase_rad = np.radians(phase_deg)

# Generate Time Base and Ideal Reference
t, fs = generate_time_array(f0=f0, num_periods=num_periods, samples_per_period=samples_per_period)
ideal_signal = generate_ideal_signal(
    selected_waveform, t, f0, amplitude,
    phase_offset_rad=phase_rad,
    dc_offset=dc_offset,
    duty_cycle=duty_cycle
)

# Compute Fourier Series Coefficients and Partial Sum
a0, a_coeffs, b_coeffs = get_coefficients_by_waveform(
    selected_waveform, amplitude, n_harmonics, duty_cycle=duty_cycle
)
reconstructed_signal, harmonics_matrix = compute_fourier_series(
    t, f0, a0, a_coeffs, b_coeffs,
    phase_offset_rad=phase_rad,
    dc_offset=dc_offset,
    return_harmonics=True
)

# Compute Quantitative Metrics
rmse = calculate_rmse(ideal_signal, reconstructed_signal)
mae = calculate_mae(ideal_signal, reconstructed_signal)
max_err = calculate_max_error(ideal_signal, reconstructed_signal)
ser_db = calculate_signal_to_error_ratio(ideal_signal, reconstructed_signal)
energy_share = calculate_parseval_power_share(selected_waveform, amplitude, a0, a_coeffs, b_coeffs)

# Metrics Grid
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    render_kpi("Harmonics Included", f"{n_harmonics}", f"Fund f₀ = {f0:.2f} Hz", "#00f0ff")
with m_col2:
    render_kpi("RMS Error (RMSE)", f"{rmse:.4f}", "Lower is better", "#a855f7")
with m_col3:
    render_kpi("Mean Absolute Error", f"{mae:.4f}", "Mean pointwise error", "#38bdf8")
with m_col4:
    render_kpi("Signal-to-Error", f"{ser_db:.1f} dB", "Reconstruction fidelity", "#10b981")
with m_col5:
    render_kpi("Parseval Energy", f"{energy_share:.1f}%", "Signal power captured", "#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Primary Interactive Time Domain Chart
fig_time = plot_time_domain(
    t=t,
    ideal=ideal_signal,
    reconstructed=reconstructed_signal,
    harmonics_matrix=harmonics_matrix,
    show_harmonics=show_harmonics,
    max_harmonic_overlays=max_overlays,
    title=f"Time Domain Reconstruction: {selected_waveform} (N={n_harmonics}, T = {1.0/f0:.3f} s)"
)
st.plotly_chart(fig_time, use_container_width=True)

# Gibbs Phenomenon and Discontinuity Deep-Dive
tab_gibbs, tab_conv, tab_anim, tab_export = st.tabs([
    "🔬 Gibbs Phenomenon",
    "📈 Convergence Curves",
    "▶ Harmonic Synthesis Animation",
    "💾 Export Data & Report"
])

with tab_gibbs:
    st.markdown("### 🔬 Gibbs Phenomenon & Discontinuity Analysis")
    if "square" in selected_waveform.lower() or "sawtooth" in selected_waveform.lower():
        gibbs = compute_gibbs_overshoot(amplitude, reconstructed_signal)
        
        g_c1, g_c2 = st.columns([1.2, 1.8])
        with g_c1:
            st.markdown(f"""
            <div class="eng-card">
                <h4 style="color: #f43f5e; margin-top: 0;">Gibbs Phenomenon Metrics</h4>
                <p style="color: #94a3b8; font-size: 0.88rem;">
                    Observed for signals with finite jump discontinuities.
                </p>
                <ul style="color: #e2e8f0; font-size: 0.88rem; line-height: 1.6;">
                    <li><strong>Nominal Amplitude (A):</strong> {amplitude:.2f}</li>
                    <li><strong>Reconstructed Peak:</strong> {gibbs['actual_max']:.4f}</li>
                    <li><strong>Empirical Overshoot:</strong> {gibbs['overshoot_percent']:.2f}%</li>
                    <li><strong>Theoretical Wilbraham Limit:</strong> ~{gibbs['theoretical_overshoot_pct']:.2f}% (1.179 × A)</li>
                    <li><strong>Max Absolute Error:</strong> {max_err:.4f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="callout-info">
                <strong>Why does overshoot persist?</strong><br>
                The Fourier series converges in the mean-square (L²) sense, not uniformly.
                Near a jump discontinuity of height 2A, the partial sum overshoots by approximately 
                <strong>8.95%</strong> of the jump (17.9% above A). As N → ∞, the width of the ripple
                collapses toward zero, but the peak overshoot magnitude does not disappear.
            </div>
            """, unsafe_allow_html=True)

        with g_c2:
            # Center close-up around the first discontinuity at t = T/2 or T
            period = 1.0 / f0
            discontinuity_time = period * 0.5
            window_sec = period * 0.15
            fig_gibbs = plot_gibbs_zoom(t, ideal_signal, reconstructed_signal, discontinuity_time, window_sec)
            st.plotly_chart(fig_gibbs, use_container_width=True)
    else:
        st.markdown(f"""
        <div class="eng-card">
            <h4 style="color: #10b981; margin-top: 0;">Continuous Waveform: No Gibbs Phenomenon</h4>
            <p style="color: #cbd5e1; font-size: 0.92rem;">
                The selected waveform (<strong>{selected_waveform}</strong>) is continuous across time without jump discontinuities.
                Therefore, the Fourier series converges uniformly, and <strong>no Gibbs overshoot occurs</strong>.
                Notice how the maximum error decays smoothly towards zero as N increases.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_conv:
    st.markdown("### 📈 Reconstruction Error vs Number of Harmonics")
    st.markdown("""
    This dynamic analysis calculates RMSE, MAE, and Peak Error as harmonic count increases up to the current setting.
    """)
    with st.spinner("Calculating convergence trajectories..."):
        max_h_conv = min(60, max(15, n_harmonics))
        df_conv = compute_convergence_data(selected_waveform, amplitude, f0, max_harmonics=max_h_conv, duty_cycle=duty_cycle)
        fig_conv = plot_error_convergence(df_conv)
        st.plotly_chart(fig_conv, use_container_width=True)
        
        st.markdown(f"""
        <div class="callout-info">
            <strong>Key Academic Observation:</strong><br>
            • <strong>RMSE and MAE</strong> drop monotonically with increasing N, confirming L² convergence (Parseval's theorem).<br>
            • For discontinuous waveforms, <strong>Max Absolute Error</strong> flattens out around the Gibbs peak value ({amplitude * 1.179:.3f}), illustrating non-uniform convergence.
        </div>
        """, unsafe_allow_html=True)

with tab_anim:
    st.markdown("### ▶ Progressive Harmonic Synthesis Animation")
    st.markdown("""
    Watch how each successive sinusoidal harmonic constructively reshapes the waveform from a single sinusoid
    into the target periodic signal.
    """)
    # Compute partial sums for animation
    step_list = []
    if n_harmonics <= 15:
        step_list = list(range(1, n_harmonics + 1))
    else:
        step_list = sorted(list(set([1, 2, 3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50, n_harmonics])))
        step_list = [s for s in step_list if s <= n_harmonics]

    partial_sums_list = []
    for s in step_list:
        a0_s, an_s, bn_s = get_coefficients_by_waveform(selected_waveform, amplitude, s, duty_cycle=duty_cycle)
        recon_s, _ = compute_fourier_series(t, f0, a0_s, an_s, bn_s, phase_offset_rad=phase_rad, dc_offset=dc_offset)
        partial_sums_list.append(recon_s)

    fig_anim = create_harmonic_building_animation(t, ideal_signal, partial_sums_list, step_list)
    st.plotly_chart(fig_anim, use_container_width=True)

with tab_export:
    st.markdown("### 💾 Export Simulation Data & Academic Summary")
    st.markdown("Download computed numerical datasets as CSV files for laboratory records or grading.")

    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        csv_waveforms = export_waveforms_csv(t, ideal_signal, reconstructed_signal)
        st.download_button(
            label="📥 Download Time-Domain Waveforms (CSV)",
            data=csv_waveforms,
            file_name=f"fourierlab_{selected_waveform.lower().replace(' ', '_')}_N{n_harmonics}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("Contains columns: time_seconds, ideal_reference, fourier_reconstructed, approximation_error.")

    with col_exp2:
        params_dict = {
            "waveform": selected_waveform,
            "n_harmonics": n_harmonics,
            "f0": f0,
            "amplitude": amplitude,
            "phase_deg": phase_deg,
            "phase_rad": phase_rad,
            "dc_offset": dc_offset,
            "num_periods": num_periods,
            "fs": fs,
            "total_samples": len(t)
        }
        metrics_dict = {
            "rmse": rmse,
            "mae": mae,
            "max_error": max_err,
            "ser_db": ser_db,
            "energy_share_pct": energy_share
        }
        student_info = st.session_state.get("student_info", {})
        summary_txt = generate_summary_report(params_dict, metrics_dict, student_info)

        st.download_button(
            label="📥 Download Full Simulation Report (TXT)",
            data=summary_txt,
            file_name=f"fourierlab_report_N{n_harmonics}.txt",
            mime="text/plain",
            use_container_width=True
        )
        st.caption("Complete text report including parameters, metrics, physical interpretation, and student details.")
