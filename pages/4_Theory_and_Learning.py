"""
Theory and Learning Laboratory
Comprehensive Signals and Systems curriculum containing 16 core theoretical topics,
step-by-step integral derivations, and 5 interactive guided experiments.
"""

import streamlit as st
import numpy as np

from src.ui_theme import apply_custom_css, render_header, render_kpi
from src.fourier_series import get_coefficients_by_waveform, compute_fourier_series, compute_gibbs_overshoot
from src.signal_generator import generate_time_array, generate_ideal_signal
from src.metrics import calculate_rmse, calculate_mae, calculate_parseval_power_share
from src.frequency_analysis import compute_single_sided_fft, find_spectral_peaks
from src.plotting import plot_time_domain, plot_gibbs_zoom, plot_frequency_spectrum

st.set_page_config(
    page_title="Theory & Learning — FourierLab",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="Theory & Learning — Signals and Systems",
    subtitle="Interactive Signals & Systems textbook: comprehensive mathematical foundations, worked derivations, and guided laboratory experiments.",
    badge_text="Signals & Systems Textbook & Interactive Labs",
    badge_color="purple"
)

tab_theory, tab_experiments, tab_worked = st.tabs([
    "📚 Complete Theory Curriculum (16 Topics)",
    "🧪 Guided Laboratory Experiments (A-E)",
    "✍️ Step-by-Step Worked Derivation"
])

with tab_theory:
    st.markdown("### 📘 Foundations of Signals and Fourier Analysis")
    
    with st.expander("1. What is a Signal?", expanded=True):
        st.markdown("""
        A **signal** is formally defined as a mathematical function of one or more independent variables 
        (most commonly time $t$) that conveys information about the state or behavior of a physical system.
        Examples include:
        - **Electrical:** Voltage $v(t)$ across a resistor, current $i(t)$ through an inductor.
        - **Acoustic:** Sound pressure variations $p(t)$ vibrating a microphone diaphragm.
        - **Biomedical:** Electrocardiogram (ECG) tracking cardiac depolarization.
        """)

    with st.expander("2. Continuous-Time vs Discrete-Time Signals"):
        st.markdown("""
        - **Continuous-Time (CT) Signal $x(t)$:** Defined for every continuous real instant $t \\in \\mathbb{R}$. 
          Physical analog signals are continuous in nature.
        - **Discrete-Time (DT) Signal $x[n]$:** Defined only at discrete integer index points $n \\in \\mathbb{Z}$, 
          obtained by sampling an analog signal at uniform intervals $T_s$:
          $$x[n] = x(n T_s)$$
        In this software laboratory, signals are represented as dense discrete numerical arrays ($t_k = k / f_s$) 
        with high sampling density to accurately approximate continuous behavior.
        """)

    with st.expander("3. Periodic Signals and Fundamental Period"):
        st.markdown("""
        A continuous-time signal $x(t)$ is **periodic** if there exists a positive real constant $T > 0$ such that:
        $$x(t + T) = x(t) \\quad \\forall t$$
        The smallest positive value of $T$ satisfying this property is called the **fundamental period** ($T_0$).
        The reciprocal gives the **fundamental frequency**:
        $$f_0 = \\frac{1}{T_0} \\text{ (in Hertz or cycles/second)}$$
        The **fundamental angular frequency** is:
        $$\\omega_0 = 2\\pi f_0 = \\frac{2\\pi}{T_0} \\text{ (in radians/second)}$$
        """)

    with st.expander("4. Sinusoids and Angular Frequency"):
        st.markdown("""
        A pure sinusoid is the most fundamental building block in linear system theory:
        $$x(t) = A \\cos(\\omega_0 t + \\phi) = A \\cos(2\\pi f_0 t + \\phi)$$
        - $A$: Peak Amplitude
        - $\\omega_0$: Angular Frequency (rad/s)
        - $\\phi$: Initial Phase Angle (radians)
        Sinusoids are **eigenfunctions** of Linear Time-Invariant (LTI) systems: passing a sinusoid through an 
        LTI system changes only its amplitude and phase, never its frequency.
        """)

    with st.expander("5. Why Fourier Series? The Principle of Superposition"):
        st.markdown("""
        Jean-Baptiste Joseph Fourier established in 1807 that **any physically realizable periodic signal** 
        can be synthesized as an infinite linear combination of harmonically related sines and cosines.
        
        **Why is this revolutionary?**
        1. Complicated differential equations governing circuit or mechanical behavior are easy to solve for sinusoids.
        2. By decomposing an arbitrary periodic input into sinusoids, solving for each harmonic individually, 
           and applying **superposition**, the complete system response is obtained effortlessly.
        """)

    with st.expander("6. Fundamental Frequency vs Harmonics"):
        st.markdown("""
        - **Fundamental Component ($n = 1$):** A sinusoid with frequency $f_1 = f_0$ and period $T$.
        - **Harmonics ($n = 2, 3, 4, \\dots$):** Sinusoids whose frequencies are strict integer multiples of the fundamental:
          $$f_n = n \\cdot f_0, \\quad \\omega_n = n \\cdot \\omega_0$$
        Because each harmonic completes exactly $n$ cycles within one fundamental period $T$, 
        their linear sum remains strictly periodic with period $T$.
        """)

    with st.expander("7. The Trigonometric Fourier Series Form"):
        st.markdown("""
        The trigonometric Fourier series represents a periodic signal $x(t)$ with period $T$ as:
        $$x_N(t) = \\frac{a_0}{2} + \\sum_{n=1}^{N} \\left[ a_n \\cos(n \\omega_0 t) + b_n \\sin(n \\omega_0 t) \\right]$$
        Alternatively, in polar form:
        $$x_N(t) = c_0 + \\sum_{n=1}^{N} c_n \\cos(n \\omega_0 t - \\theta_n)$$
        where $c_0 = a_0 / 2$, $c_n = \\sqrt{a_n^2 + b_n^2}$, and $\\theta_n = \\operatorname{atan2}(b_n, a_n)$.
        """)

    with st.expander("8. Fourier Coefficient Integral Formulas"):
        st.markdown("""
        Using the mutual orthogonality of sinusoidal functions over a period $T$, the coefficients are determined by:
        
        $$\\mathbf{a_0} = \\frac{2}{T} \\int_{t_0}^{t_0 + T} x(t) \\, dt \\quad \\implies \\quad \\frac{a_0}{2} = \\frac{1}{T} \\int_{0}^{T} x(t) \\, dt \\text{ (Average DC Value)}$$
        
        $$\\mathbf{a_n} = \\frac{2}{T} \\int_{t_0}^{t_0 + T} x(t) \\cos(n \\omega_0 t) \\, dt \\quad (n = 1, 2, 3, \\dots)$$
        
        $$\\mathbf{b_n} = \\frac{2}{T} \\int_{t_0}^{t_0 + T} x(t) \\sin(n \\omega_0 t) \\, dt \\quad (n = 1, 2, 3, \\dots)$$
        
        *Note on Convention:* Writing $a_0/2$ in the series ensures the formula for $a_n$ holds uniformly for $n=0$.
        """)

    with st.expander("9. Square-Wave Reconstruction"):
        st.markdown("""
        An ideal square wave with zero mean and odd symmetry $x(-t) = -x(t)$ has:
        - $a_0 = 0$ (zero DC mean)
        - $a_n = 0$ for all $n$ (even cosine functions integrate to zero against an odd signal)
        - $b_n = \\frac{4A}{n\\pi}$ for odd $n$; $b_n = 0$ for even $n$
        
        $$\\mathbf{x_N(t)} = \\frac{4A}{\\pi} \\left[ \\sin(\\omega_0 t) + \\frac{1}{3}\\sin(3\\omega_0 t) + \\frac{1}{5}\\sin(5\\omega_0 t) + \\dots \\right]$$
        """)

    with st.expander("10. Sawtooth-Wave Reconstruction"):
        st.markdown("""
        A rising linear ramp centered with odd symmetry over $(-T/2, T/2)$:
        - $a_0 = 0, \\quad a_n = 0$
        - $b_n = \\frac{2A}{n\\pi} (-1)^{n+1}$
        
        $$\\mathbf{x_N(t)} = \\frac{2A}{\\pi} \\sum_{n=1}^{N} \\frac{(-1)^{n+1}}{n} \\sin(n \\omega_0 t)$$
        *Notice:* **All integer harmonics** (both even and odd) are present because the sawtooth lacks half-wave symmetry.
        """)

    with st.expander("11. Triangular-Wave Reconstruction"):
        st.markdown("""
        A continuous triangular wave with peak amplitude $A$ and odd symmetry:
        - $a_0 = 0, \\quad a_n = 0$
        - $b_n = \\frac{8A}{\\pi^2 n^2} (-1)^{(n-1)/2}$ for odd $n$; $b_n = 0$ for even $n$
        
        $$\\mathbf{x_N(t)} = \\frac{8A}{\\pi^2} \\left[ \\sin(\\omega_0 t) - \\frac{1}{9}\\sin(3\\omega_0 t) + \\frac{1}{25}\\sin(5\\omega_0 t) - \\dots \\right]$$
        Harmonics decay as $1/n^2$ ($-40$ dB/dec), converging much faster than square or sawtooth waves.
        """)

    with st.expander("12. Harmonic Convergence & Parseval's Theorem"):
        st.markdown("""
        **Parseval's Identity** states that the average power of a periodic signal equals the sum of the powers of its individual Fourier components:
        
        $$P = \\frac{1}{T} \\int_{0}^{T} |x(t)|^2 \\, dt = \\left(\\frac{a_0}{2}\\right)^2 + \\frac{1}{2} \\sum_{n=1}^{\\infty} (a_n^2 + b_n^2)$$
        
        This guarantees that the total energy captured by $N$ harmonics approaches 100% as $N \\to \\infty$.
        """)

    with st.expander("13. The Gibbs Phenomenon"):
        st.markdown("""
        At points of jump discontinuity (such as square or sawtooth step transitions), partial Fourier sums exhibit 
        an oscillatory ringing overshoot.
        - **Wilbraham-Gibbs Peak:** The maximum overshoot approaches approximately **8.95% of the total jump height** 
          ($\\approx 17.9\\%$ above the flat-top amplitude $A$).
        - **Limiting Behavior:** As $N \\to \\infty$, the ripple compresses in width toward the discontinuity point, 
          but the peak overshoot does **not** vanish. Convergence is non-uniform at the discontinuity.
        """)

    with st.expander("14. Time-Domain vs Frequency-Domain Duality"):
        st.markdown("""
        - **Time Domain $x(t)$:** Shows *when* events happen and the waveform's instantaneous voltage.
        - **Frequency Domain $X(f)$:** Shows *which* frequencies are present and their relative amplitudes/phases.
        They are complementary views of the same physical reality, linked by the Fourier Transform.
        """)

    with st.expander("15. Sampling Theorem & Discrete FFT Basics"):
        st.markdown("""
        - **Nyquist-Shannon Theorem:** To reconstruct a continuous band-limited signal without aliasing, 
          the sampling rate $f_s$ must strictly exceed twice the highest frequency present:
          $$f_s > 2 f_{\\max}$$
        - **Fast Fourier Transform (FFT):** An efficient $O(N \\log N)$ algorithm to compute the Discrete Fourier Transform (DFT).
        - **Frequency Resolution:** $\\Delta f = f_s / N_{\\text{samples}} = 1 / T_{\\text{acquisition}}$.
        """)

    with st.expander("16. Practical Real-World Applications"):
        st.markdown("""
        1. **Telecommunications:** Frequency-Division Multiplexing (FDM), modulating voice/data onto carrier frequencies.
        2. **Audio & Music Synthesis:** Additive synthesis in synthesizers creates complex instrument timbres by summing sine waves.
        3. **Power Quality Engineering:** Identifying harmonic distortion caused by non-linear industrial loads (rectifiers, inverters).
        4. **Structural Vibration Analysis:** Detecting natural resonance frequencies in bridges, aircraft wings, and turbine blades.
        """)

with tab_experiments:
    st.markdown("### 🧪 Guided Interactive Laboratory Experiments")
    st.markdown("Conduct the 5 core Signals & Systems laboratory experiments with live simulations and observations.")

    exp_choice = st.selectbox(
        "Select Experiment",
        [
            "Experiment A: Effect of Harmonic Count",
            "Experiment B: Gibbs Phenomenon & Discontinuity Zoom",
            "Experiment C: Fundamental Frequency Variation",
            "Experiment D: Waveform Comparison & Decay Rates",
            "Experiment E: Time-Domain vs Frequency-Domain Dual Inspection"
        ]
    )

    if "Experiment A" in exp_choice:
        st.markdown("""
        #### 🎯 Experiment A: Effect of Harmonic Count on Signal Fidelity
        **Objective:** Investigate how increasing the number of harmonics $N$ affects waveform shape and RMSE error.
        """)
        c_a1, c_a2 = st.columns([1, 2])
        with c_a1:
            exp_a_n = st.slider("Harmonics (N)", 1, 40, 5, key="exp_a_n")
            exp_a_wf = st.selectbox("Waveform", ["Square Wave", "Sawtooth Wave", "Triangular Wave"], key="exp_a_wf")
            
            t_a, fs_a = generate_time_array(1.0, 2.0, 1000)
            id_a = generate_ideal_signal(exp_a_wf, t_a, 1.0, 1.0)
            a0_a, an_a, bn_a = get_coefficients_by_waveform(exp_a_wf, 1.0, exp_a_n)
            rec_a, _ = compute_fourier_series(t_a, 1.0, a0_a, an_a, bn_a)
            rmse_a = calculate_rmse(id_a, rec_a)
            pwr_a = calculate_parseval_power_share(exp_a_wf, 1.0, a0_a, an_a, bn_a)

            render_kpi("Recorded RMSE", f"{rmse_a:.4f}", "Error drops as N rises", "#00f0ff")
            render_kpi("Energy Captured", f"{pwr_a:.2f}%", "Parseval theorem", "#10b981")

        with c_a2:
            fig_a = plot_time_domain(t_a, id_a, rec_a, title=f"Exp A: {exp_a_wf} with N={exp_a_n} (RMSE={rmse_a:.4f})")
            st.plotly_chart(fig_a, use_container_width=True)

        st.markdown("""
        <div class="callout-success">
            <strong>Lab Takeaway:</strong> With small N (e.g., N=1), the reconstruction is a crude single sinusoid. 
            As N increases, higher-frequency components sharpen the slopes and corner transitions, driving RMSE toward zero.
        </div>
        """, unsafe_allow_html=True)

    elif "Experiment B" in exp_choice:
        st.markdown("""
        #### 🎯 Experiment B: Investigating Gibbs Phenomenon Near Discontinuities
        **Objective:** Observe the persistence of overshoot near jump discontinuities as $N$ increases.
        """)
        c_b1, c_b2 = st.columns([1, 2])
        with c_b1:
            exp_b_n = st.slider("Harmonics (N)", 3, 50, 15, key="exp_b_n")
            t_b, fs_b = generate_time_array(1.0, 1.0, 2000)
            id_b = generate_ideal_signal("Square Wave", t_b, 1.0, 1.0)
            a0_b, an_b, bn_b = get_coefficients_by_waveform("Square Wave", 1.0, exp_b_n)
            rec_b, _ = compute_fourier_series(t_b, 1.0, a0_b, an_b, bn_b)
            gibbs_b = compute_gibbs_overshoot(1.0, rec_b)

            render_kpi("Peak Overshoot", f"{gibbs_b['overshoot_percent']:.2f}%", "Above nominal amplitude 1.0", "#f43f5e")
            render_kpi("Observed Peak", f"{gibbs_b['actual_max']:.4f}", "Theoretical: ~1.179", "#f59e0b")

        with c_b2:
            fig_b = plot_gibbs_zoom(t_b, id_b, rec_b, discontinuity_time=0.5, window_sec=0.12)
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("""
        <div class="callout-warning">
            <strong>Lab Takeaway:</strong> Notice that increasing N does NOT eliminate the peak overshoot (~17.9% above A). 
            Instead, the ripples become narrower and pack tightly against the jump boundary at t = 0.5 s.
        </div>
        """, unsafe_allow_html=True)

    elif "Experiment C" in exp_choice:
        st.markdown("""
        #### 🎯 Experiment C: Fundamental Frequency Variation & Period Scaling
        **Objective:** Demonstrate the reciprocal relationship $T = 1/f_0$ and harmonic frequency scaling $f_n = n f_0$.
        """)
        exp_c_f0 = st.slider("Fundamental Frequency f₀ (Hz)", 0.5, 10.0, 2.0, 0.5, key="exp_c_f0")
        t_c, fs_c = generate_time_array(exp_c_f0, num_periods=3.0, samples_per_period=1000)
        id_c = generate_ideal_signal("Sawtooth Wave", t_c, exp_c_f0, 1.0)
        a0_c, an_c, bn_c = get_coefficients_by_waveform("Sawtooth Wave", 1.0, 10)
        rec_c, _ = compute_fourier_series(t_c, exp_c_f0, a0_c, an_c, bn_c)

        col_c_k1, col_c_k2, col_c_k3 = st.columns(3)
        with col_c_k1:
            render_kpi("Period (T)", f"{1.0/exp_c_f0:.3f} s", "T = 1 / f₀", "#00f0ff")
        with col_c_k2:
            render_kpi("Harmonic 3 Freq", f"{3 * exp_c_f0:.1f} Hz", "f₃ = 3 · f₀", "#a855f7")
        with col_c_k3:
            render_kpi("Harmonic 5 Freq", f"{5 * exp_c_f0:.1f} Hz", "f₅ = 5 · f₀", "#10b981")

        fig_c = plot_time_domain(t_c, id_c, rec_c, title=f"Exp C: Sawtooth at f₀ = {exp_c_f0:.1f} Hz (Period = {1.0/exp_c_f0:.3f} s)")
        st.plotly_chart(fig_c, use_container_width=True)

    elif "Experiment D" in exp_choice:
        st.markdown("""
        #### 🎯 Experiment D: Waveform Comparison & Decay Rates (1/n vs 1/n²)
        **Objective:** Compare how continuity affects the convergence rate and harmonic energy distribution.
        """)
        t_d, _ = generate_time_array(1.0, 1.5, 1000)
        
        # Compare Square vs Triangle at N=5
        a0_sq, an_sq, bn_sq = get_coefficients_by_waveform("Square Wave", 1.0, 5)
        rec_sq, _ = compute_fourier_series(t_d, 1.0, a0_sq, an_sq, bn_sq)
        pwr_sq = calculate_parseval_power_share("Square Wave", 1.0, a0_sq, an_sq, bn_sq)

        a0_tr, an_tr, bn_tr = get_coefficients_by_waveform("Triangular Wave", 1.0, 5)
        rec_tr, _ = compute_fourier_series(t_d, 1.0, a0_tr, an_tr, bn_tr)
        pwr_tr = calculate_parseval_power_share("Triangular Wave", 1.0, a0_tr, an_tr, bn_tr)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            render_kpi("Square Wave (Discontinuous)", f"{pwr_sq:.1f}% Power", "1/n decay (slow)", "#f59e0b")
            fig_d1 = plot_time_domain(t_d, generate_ideal_signal("Square Wave", t_d, 1.0, 1.0), rec_sq, title="Square Wave (N=5)")
            st.plotly_chart(fig_d1, use_container_width=True)

        with col_d2:
            render_kpi("Triangular Wave (Continuous)", f"{pwr_tr:.2f}% Power", "1/n² decay (rapid)", "#10b981")
            fig_d2 = plot_time_domain(t_d, generate_ideal_signal("Triangular Wave", t_d, 1.0, 1.0), rec_tr, title="Triangular Wave (N=5)")
            st.plotly_chart(fig_d2, use_container_width=True)

    elif "Experiment E" in exp_choice:
        st.markdown("""
        #### 🎯 Experiment E: Time vs Frequency Domain Duality
        **Objective:** Simultaneously inspect time-domain reconstruction and its corresponding single-sided FFT spectrum.
        """)
        exp_e_n = st.slider("Harmonics (N)", 1, 20, 7, key="exp_e_n")
        t_e, fs_e = generate_time_array(2.0, 4.0, 1000)
        id_e = generate_ideal_signal("Square Wave", t_e, 2.0, 1.0)
        a0_e, an_e, bn_e = get_coefficients_by_waveform("Square Wave", 1.0, exp_e_n)
        rec_e, _ = compute_fourier_series(t_e, 2.0, a0_e, an_e, bn_e)
        freqs_e, amps_e, _ = compute_single_sided_fft(rec_e, fs_e)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            fig_et = plot_time_domain(t_e, id_e, rec_e, title=f"Time Domain (N={exp_e_n})")
            st.plotly_chart(fig_et, use_container_width=True)
        with col_e2:
            fig_ef = plot_frequency_spectrum(freqs_e, amps_e, freqs_e, amps_e, f0=2.0, max_freq_display=40.0)
            st.plotly_chart(fig_ef, use_container_width=True)

with tab_worked:
    st.markdown("### ✍️ Step-by-Step Analytical Derivation: Ideal Square Wave")
    st.markdown("""
    This worked example provides the exact analytical proof and integration steps for an ideal square wave, 
    frequently asked during Signals & Systems viva examinations.
    """)

    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #00f0ff; margin-top: 0;">Step 1: Signal Definition & Symmetry Analysis</h4>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6;">
            Consider a periodic square wave $x(t)$ with period $T$, fundamental angular frequency $\\omega_0 = 2\\pi/T$, 
            and peak amplitude $A$:
        </p>
        <div style="background: rgba(15,23,42,0.8); padding: 0.75rem; border-radius: 6px; font-family: monospace; color: #38bdf8;">
            x(t) = +A, &nbsp; for 0 &lt; t &lt; T/2<br>
            x(t) = -A, &nbsp; for T/2 &lt; t &lt; T
        </div>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
            <strong>Symmetry Check:</strong><br>
            1. <strong>Half-Wave Symmetry:</strong> $x(t + T/2) = -x(t)$. This guarantees all even harmonics are identically zero ($a_{2k} = b_{2k} = 0$).<br>
            2. <strong>Odd Symmetry:</strong> $x(-t) = -x(t)$. This guarantees all cosine coefficients are zero ($a_n = 0$ for all $n$).
        </p>
    </div>

    <div class="eng-card">
        <h4 style="color: #a855f7; margin-top: 0;">Step 2: Average DC Offset (a₀)</h4>
        <div style="background: rgba(15,23,42,0.8); padding: 0.75rem; border-radius: 6px; font-family: monospace; color: #f1f5f9;">
            a₀ = (2/T) ∫₀ᵀ x(t) dt = (2/T) [ ∫₀^{T/2} (+A) dt + ∫_{T/2}^{T} (-A) dt ]<br>
            a₀ = (2/T) [ A(T/2) - A(T - T/2) ] = (2/T) [ AT/2 - AT/2 ] = 0
        </div>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.5rem;">
            Therefore, the DC component $a_0/2 = 0$.
        </p>
    </div>

    <div class="eng-card">
        <h4 style="color: #10b981; margin-top: 0;">Step 3: Sine Coefficients (bₙ) Integration</h4>
        <div style="background: rgba(15,23,42,0.8); padding: 0.75rem; border-radius: 6px; font-family: monospace; color: #f1f5f9; line-height: 1.8;">
            b_n = (2/T) ∫₀ᵀ x(t) sin(n ω₀ t) dt<br>
            b_n = (2/T) [ ∫₀^{T/2} A sin(n ω₀ t) dt - ∫_{T/2}^{T} A sin(n ω₀ t) dt ]<br>
            b_n = (2A / T) [ -cos(n ω₀ t) / (n ω₀) ]₀^{T/2} - (2A / T) [ -cos(n ω₀ t) / (n ω₀) ]_{T/2}^{T}
        </div>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
            Substitute $\\omega_0 T = 2\\pi$, so $\\omega_0 (T/2) = n\\pi$:
        </p>
        <div style="background: rgba(15,23,42,0.8); padding: 0.75rem; border-radius: 6px; font-family: monospace; color: #38bdf8; line-height: 1.8;">
            b_n = (2A / 2πn) [ (1 - cos(nπ)) - (cos(nπ) - cos(2nπ)) ]<br>
            b_n = (A / πn) [ 1 - cos(nπ) - cos(nπ) + 1 ] = (2A / πn) [ 1 - cos(nπ) ]
        </div>
        <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.6; margin-top: 0.5rem;">
            Evaluate for even and odd $n$:
        </p>
        <ul style="color: #e2e8f0; font-size: 0.88rem; line-height: 1.8;">
            <li>If $n$ is <strong>even</strong>: $\\cos(n\\pi) = +1 \\implies 1 - 1 = 0 \\implies \\mathbf{b_n = 0}$.</li>
            <li>If $n$ is <strong>odd</strong>: $\\cos(n\\pi) = -1 \\implies 1 - (-1) = 2 \\implies \\mathbf{b_n = \\frac{4A}{n\\pi}}$.</li>
        </ul>
    </div>

    <div class="eng-card">
        <h4 style="color: #f59e0b; margin-top: 0;">Step 4: Final Synthesized Series</h4>
        <div style="background: rgba(15,23,42,0.9); padding: 1rem; border-radius: 6px; font-family: monospace; color: #00f0ff; font-size: 1.05rem; text-align: center;">
            x_N(t) = (4A / π) · Σ_{k=1,3,5...}^N [ sin(k ω₀ t) / k ]
        </div>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.75rem;">
            First three non-zero terms: x(t) ≈ (4A/π) sin(ω₀t) + (4A/3π) sin(3ω₀t) + (4A/5π) sin(5ω₀t).
        </p>
    </div>
    """, unsafe_allow_html=True)
