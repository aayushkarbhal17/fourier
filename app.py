"""
FourierLab — Interactive Fourier Series Visualizer
Signals and Systems Mini Project
Main Entry Point & Interactive Dashboard
"""

import streamlit as st
import numpy as np

from src.ui_theme import apply_custom_css, render_header, render_kpi
from src.fourier_series import get_coefficients_by_waveform, compute_fourier_series
from src.signal_generator import generate_time_array, generate_ideal_signal
from src.metrics import calculate_rmse, calculate_mae, calculate_signal_to_error_ratio, calculate_parseval_power_share
from src.plotting import plot_time_domain

# Configure page layout and metadata
st.set_page_config(
    page_title="FourierLab — Interactive Fourier Series Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom dark navy design system
apply_custom_css()

# Sidebar branding & quick navigation info
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <span class="eng-badge">ECE / EE Lab</span>
        <h2 style="font-size: 1.4rem; margin: 0.2rem 0; color: #00f0ff;">⚡ FourierLab</h2>
        <p style="font-size: 0.82rem; color: #94a3b8; margin: 0;">Signals & Systems Visualizer</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🧭 Laboratory Modules")
    st.markdown("""
    - **1. Signal Lab**: Deep-dive simulator & Gibbs analysis
    - **2. Harmonic Explorer**: Isolate & inspect individual harmonics
    - **3. Frequency Analysis**: FFT & single-sided spectrum
    - **4. Theory & Learning**: 16 textbook topics & 5 guided experiments
    - **5. Quiz & Viva**: 15+ MCQs & viva questions
    - **6. About Project**: Abstract, architecture & submission dossier
    """)
    st.markdown("---")
    st.info("💡 **Tip:** Use the navigation menu above to access dedicated laboratory pages.")

# Hero Banner
render_header(
    title="FourierLab — Interactive Fourier Series Visualizer",
    subtitle="A comprehensive Signals & Systems laboratory demonstration showing how complex periodic signals are synthesized from harmonically related sinusoids.",
    badge_text="Signals & Systems Mini Project • Academic Demonstration",
    badge_color="cyan"
)

# Executive KPI Grid
col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi("Core Principle", "Trigonometric Series", "x_N(t) = a0/2 + Σ [an cos + bn sin]", "#00f0ff")
with col2:
    render_kpi("Harmonics Range", "1 to 100 Harmonics", "Real-time vector synthesis", "#a855f7")
with col3:
    render_kpi("Spectral Engine", "FFT & Analytical", "Single-sided scaled spectrum", "#10b981")
with col4:
    render_kpi("Fidelity Assessment", "RMSE & Parseval", "Quantitative convergence metrics", "#f59e0b")

st.markdown("<br>", unsafe_allow_html=True)

# Quick-Start Interactive Laboratory
st.markdown("""
<div class="eng-card">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem;">
        <h3 style="margin: 0; color: #00f0ff; font-size: 1.25rem;">🚀 Quick-Start Simulation</h3>
        <span class="eng-badge eng-badge-purple">Live Interactive Preview</span>
    </div>
    <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
        Adjust the waveform type and harmonic count below to witness constructive Fourier reconstruction in real time.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Controls
c_wf, c_harm, c_f0, c_amp = st.columns([1.5, 2, 1.2, 1.2])
with c_wf:
    quick_wf = st.selectbox(
        "Waveform Type",
        ["Square Wave", "Sawtooth Wave", "Triangular Wave", "Sine Wave", "Pulse Train (50% Duty)"],
        index=0
    )
with c_harm:
    quick_n = st.slider("Number of Harmonics (N)", min_value=1, max_value=50, value=7, step=1)
with c_f0:
    quick_f0 = st.number_input("Fundamental f₀ (Hz)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
with c_amp:
    quick_amp = st.number_input("Amplitude (A)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

# Quick Computation
duty_val = 0.5
t_quick, fs_quick = generate_time_array(f0=quick_f0, num_periods=2.0, samples_per_period=1000)
ideal_quick = generate_ideal_signal(quick_wf, t_quick, quick_f0, quick_amp, duty_cycle=duty_val)
a0_q, an_q, bn_q = get_coefficients_by_waveform(quick_wf, quick_amp, quick_n, duty_cycle=duty_val)
recon_quick, _ = compute_fourier_series(t_quick, quick_f0, a0_q, an_q, bn_q)

# Compute quick metrics
rmse_q = calculate_rmse(ideal_quick, recon_quick)
mae_q = calculate_mae(ideal_quick, recon_quick)
ser_q = calculate_signal_to_error_ratio(ideal_quick, recon_quick)
energy_q = calculate_parseval_power_share(quick_wf, quick_amp, a0_q, an_q, bn_q)

# Metric readout row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Root Mean Square Error", f"{rmse_q:.4f}")
with m2:
    st.metric("Mean Absolute Error", f"{mae_q:.4f}")
with m3:
    st.metric("Signal-to-Error Ratio", f"{ser_q:.2f} dB")
with m4:
    st.metric("Parseval Energy Share", f"{energy_q:.1f}%")

# Plot Quick Preview Chart
fig_quick = plot_time_domain(
    t=t_quick,
    ideal=ideal_quick,
    reconstructed=recon_quick,
    title=f"Fourier Reconstruction: {quick_wf} with N={quick_n} Harmonics (f₀ = {quick_f0} Hz)"
)
st.plotly_chart(fig_quick, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Feature Highlights & Navigation Cards
st.markdown("### 📚 Explore Project Capabilities")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #00f0ff; margin-top: 0;">🔬 1. Signal Lab</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            Full parameter customization: DC offset, phase shifts, harmonic overlays, Gibbs phenomenon analysis,
            and frame-by-frame harmonic synthesis animation.
        </p>
        <span class="eng-badge">Core Simulator</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #a855f7; margin-top: 0;">📖 4. Theory & Learning</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            16 comprehensive textbook topics with LaTeX formulas, step-by-step integral derivations,
            and 5 interactive guided experiments.
        </p>
        <span class="eng-badge eng-badge-purple">Textbook & Labs</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #38bdf8; margin-top: 0;">🔍 2. Harmonic Explorer</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            Inspect individual sinusoids, verify orthogonality, view interactive stem plots of |cₙ|,
            and export complete harmonic coefficient tables as CSV.
        </p>
        <span class="eng-badge">Harmonic Isolation</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #10b981; margin-top: 0;">📝 5. Quiz & Viva</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            Self-assessment quiz with 15+ multiple-choice questions, instant score calculation,
            and 16+ external examiner viva questions with answers.
        </p>
        <span class="eng-badge eng-badge-green">Exam Prep</span>
    </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #f59e0b; margin-top: 0;">📊 3. Frequency Analysis</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            Single-sided FFT amplitude spectra, automated harmonic peak detection, windowing comparisons
            (Hann, Hamming, Blackman), and spectral leakage education.
        </p>
        <span class="eng-badge">Frequency Domain</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #c084fc; margin-top: 0;">🎓 6. About Project</h4>
        <p style="color: #94a3b8; font-size: 0.88rem;">
            Academic abstract, system architecture flowchart, mathematical references,
            and customizable student submission report generation.
        </p>
        <span class="eng-badge eng-badge-purple">Academic Dossier</span>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;">
    <strong>FourierLab</strong> • Signals and Systems Mini Project • Built with Python, Streamlit, Plotly, NumPy & SciPy.<br>
    All computations run 100% locally with mathematical precision.
</div>
""", unsafe_allow_html=True)
