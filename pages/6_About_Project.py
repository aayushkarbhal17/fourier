"""
About & Project Information Module
Academic documentation, student submission dossier customization,
project abstract, system architecture, methodology, advantages, and limitations.
"""

import streamlit as st
from datetime import datetime
from src.ui_theme import apply_custom_css, render_header, render_kpi

st.set_page_config(
    page_title="About & Project Information — FourierLab",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()

render_header(
    title="About & Academic Project Information",
    subtitle="Official academic documentation, system architecture, methodology, and student submission dossier generator for college viva evaluation.",
    badge_text="Academic Mini Project Documentation",
    badge_color="purple"
)

# Session state initialization for student metadata
if "student_info" not in st.session_state:
    st.session_state.student_info = {
        "name": "",
        "roll": "",
        "department": "Electronics & Communication Engineering",
        "college": "",
        "year": "2025 - 2026",
        "supervisor": ""
    }

tab_dossier, tab_academic, tab_arch = st.tabs([
    "👤 Student Submission Dossier",
    "📄 Academic Project Abstract & Report",
    "🏗️ System Architecture & Methodology"
])

with tab_dossier:
    st.markdown("### 🎓 Student Submission Metadata")
    st.markdown("""
    Enter your academic credentials below. These details will be automatically incorporated into generated 
    laboratory simulation reports, export headers, and the official submission cover sheet.
    """)

    c_form1, c_form2 = st.columns(2)
    with c_form1:
        s_name = st.text_input("Student Name", value=st.session_state.student_info.get("name", ""), placeholder="e.g. Alex Johnson")
        s_roll = st.text_input("Roll / Registration Number", value=st.session_state.student_info.get("roll", ""), placeholder="e.g. 21EC042")
        s_dept = st.text_input("Department / Branch", value=st.session_state.student_info.get("department", "Electronics & Communication Engineering"))

    with c_form2:
        s_college = st.text_input("College / University Name", value=st.session_state.student_info.get("college", ""), placeholder="e.g. National Institute of Technology")
        s_year = st.text_input("Academic Year / Semester", value=st.session_state.student_info.get("year", "2025 - 2026"))
        s_supervisor = st.text_input("Project Guide / Faculty Supervisor", value=st.session_state.student_info.get("supervisor", ""), placeholder="e.g. Prof. R. Sharma")

    if st.button("💾 Save Academic Profile"):
        st.session_state.student_info = {
            "name": s_name,
            "roll": s_roll,
            "department": s_dept,
            "college": s_college,
            "year": s_year,
            "supervisor": s_supervisor
        }
        st.success("✅ Academic profile updated successfully! All exported reports will now feature your details.")

    # Printable Preview Card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📄 Submission Dossier Preview")
    
    name_display = s_name if s_name else "Your Name Here"
    roll_display = s_roll if s_roll else "Roll No. Here"
    college_display = s_college if s_college else "Institution Name Here"
    supervisor_display = s_supervisor if s_supervisor else "Faculty Guide Here"

    dossier_card = f"""
    <div class="eng-card" style="border: 2px solid #00f0ff; padding: 2rem; background: rgba(15, 23, 42, 0.95); text-align: center;">
        <h3 style="color: #00f0ff; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem;">Mini Project Submission Dossier</h3>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem;">Department of {s_dept}</p>
        
        <h1 style="color: #f8fafc; font-size: 2.2rem; margin: 1rem 0;">FourierLab — Interactive Fourier Series Visualizer</h1>
        <p style="color: #38bdf8; font-size: 1.1rem; margin-bottom: 2rem;">A Course Mini-Project in Signals and Systems (Course Code: EC202 / EE204)</p>
        
        <div style="display: flex; justify-content: space-around; text-align: left; margin: 2rem 0; border-top: 1px solid rgba(255,255,255,0.1); border-bottom: 1px solid rgba(255,255,255,0.1); padding: 1.5rem 0;">
            <div>
                <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.2rem;">Submitted By</p>
                <h3 style="color: #f8fafc; margin: 0; font-size: 1.2rem;">{name_display}</h3>
                <p style="color: #38bdf8; margin: 0.2rem 0 0 0; font-family: monospace;">Roll No: {roll_display}</p>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.2rem;">Under the Guidance of</p>
                <h3 style="color: #f8fafc; margin: 0; font-size: 1.2rem;">{supervisor_display}</h3>
                <p style="color: #cbd5e1; margin: 0.2rem 0 0 0;">Department of {s_dept}</p>
            </div>
            <div>
                <p style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 0.2rem;">Academic Year</p>
                <h3 style="color: #f8fafc; margin: 0; font-size: 1.2rem;">{s_year}</h3>
                <p style="color: #a855f7; margin: 0.2rem 0 0 0;">{college_display}</p>
            </div>
        </div>
        
        <p style="color: #64748b; font-size: 0.8rem; margin-top: 1rem;">
            Certified that this application was developed and executed locally with numerical and academic precision.
        </p>
    </div>
    """
    st.markdown(dossier_card, unsafe_allow_html=True)

with tab_academic:
    st.markdown("### 📄 Academic Project Report Sections")

    with st.expander("1. Project Abstract", expanded=True):
        st.markdown("""
        **Abstract:**
        Fourier analysis is an indispensable pillar of modern signal processing, communications, and linear system theory. 
        However, students frequently struggle with the abstract transition between closed-form integral formulas and the physical 
        time-frequency reality of harmonic synthesis. 

        **FourierLab** is an interactive, academically rigorous, and visually dynamic engineering web dashboard designed to bridge 
        this pedagogical gap. The application allows students to simulate the trigonometric Fourier series reconstruction of canonical 
        periodic waveforms (square, sawtooth, triangular, sinusoidal, and pulse trains) up to 100 harmonics with arbitrary fundamental 
        frequencies, phase shifts, and DC offsets. 

        Crucially, the numerical engine strictly differentiates between continuous analytical reference signals and discrete partial sums, 
        enabling exact quantitative validation of Parseval's energy theorem, root-mean-square convergence rates, and the 
        Wilbraham-Gibbs phenomenon near jump discontinuities. Equipped with single-sided FFT spectral inspection, isolated harmonic 
        parameter decomposition, automated error tracking, interactive self-assessment quizzes, and downloadable datasets, FourierLab 
        serves as a comprehensive standalone mini-project suitable for university laboratory curriculum and viva voce demonstration.
        """)

    with st.expander("2. Problem Statement"):
        st.markdown("""
        In traditional Signals and Systems pedagogy, students learn Fourier series primarily through manual symbolic integration. 
        While analytically rigorous, this approach introduces severe pedagogical hurdles:
        1. **Lack of Intuition:** Students cannot visualize how adding successive high-frequency sinusoids cancels unwanted ripples 
           and sharpens edges.
        2. **Misunderstanding Gibbs Phenomenon:** Textbooks state that Fourier series converge to the average at jump discontinuities, 
           yet students rarely observe that the ~8.95% peak overshoot remains invariant as harmonic count approaches infinity.
        3. **Separation of Time and Frequency Domains:** Students treat Fourier coefficients $a_n, b_n$ as abstract algebra rather than 
           discrete spectral lines measurable via a Fast Fourier Transform.
        4. **Need for Local, Reproducible Labs:** Existing commercial simulation suites (like MATLAB) require expensive institutional 
           licenses and heavy installations, rendering them inaccessible for rapid demonstrations on student laptops.
        """)

    with st.expander("3. Objectives and Scope"):
        st.markdown("""
        - **Primary Objective:** Build an interactive, standalone, open-source software application in Python to demonstrate, 
          simulate, and validate trigonometric Fourier series reconstruction in real time.
        - **Specific Objectives:**
          1. Implement exact analytical Fourier coefficients for standard canonical periodic waveforms.
          2. Visualize constructive waveform building through animated progressive summation.
          3. Quantify reconstruction accuracy via RMSE, MAE, Max Absolute Error, and Signal-to-Error Ratio (SER).
          4. Empirically verify Parseval's energy conservation theorem.
          5. Demonstrate the Gibbs phenomenon through localized discontinuity zoom plots and overshoot metrics.
          6. Compute single-sided FFT amplitude spectra to reinforce time-frequency duality and windowing effects.
          7. Provide an integrated self-assessment quiz and oral viva preparation module.
        """)

    with st.expander("4. Advantages and Limitations"):
        st.markdown("""
        **Advantages:**
        - **100% Local & Free:** Requires zero paid cloud accounts, external APIs, or database servers. Runs entirely in Python.
        - **Academic Precision:** Reconstructed signals are synthesized from true mathematical Fourier coefficients rather than 
          merely calling black-box waveform generator functions.
        - **Interactive Plotly Visualization:** Smooth pan, zoom, hover tooltips, and high-DPI image exports.
        - **Pedagogically Complete:** Integrates theory, derivations, interactive experiments, quiz assessment, and data export in one interface.

        **Limitations:**
        - **Discrete Approximation:** While analytical formulas define the continuous series, computation on a digital computer 
          requires discrete time sampling ($t_k$), which introduces minor discretization effects at extremely high harmonic frequencies.
        - **Finite Truncation:** Calculations are evaluated up to a practical limit of $N = 100$ harmonics to preserve instantaneous 
          UI responsiveness without browser lag.
        """)

    with st.expander("5. Real-World Engineering Applications"):
        st.markdown("""
        1. **Telecommunications:** Frequency-Division Multiplexing (FDM), carrier modulation, and bandwidth allocation.
        2. **Audio & Acoustics:** Additive music synthesis (pipe organs, electronic synthesizers), audio equalization, and timbre engineering.
        3. **Power Systems:** Harmonic distortion analysis (THD) generated by switched-mode power supplies, inverters, and industrial rectifiers.
        4. **Image & Video Compression:** JPEG and MPEG standards rely on the Discrete Cosine Transform (DCT), a direct Fourier derivative.
        5. **Biomedical Engineering:** ECG and EEG rhythm decomposition, noise removal via bandpass filtering.
        """)

    with st.expander("6. Conclusion"):
        st.markdown("""
        The **FourierLab** mini-project successfully realizes a high-performance, aesthetically modern, and mathematically sound 
        educational simulation suite. By unifying analytical derivations with dynamic interactive visualizations, the project 
        transforms abstract mathematical series into an intuitive, observable, and measurable engineering reality.
        """)

with tab_arch:
    st.markdown("### 🏗️ System Architecture & Algorithmic Pipeline")
    
    st.markdown("""
    FourierLab is organized following strict separation of concerns, decoupling mathematical computations from UI rendering.
    """)

    st.markdown("""
    ```text
    ┌────────────────────────────────────────────────────────────────────────┐
    │                       FourierLab Architecture                          │
    └────────────────────────────────────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           ▼                        ▼                        ▼
    ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
    │   Math &     │         │ Frequency &  │         │ Quantitative │
    │ Coefficients │         │  FFT Engine  │         │   Metrics    │
    │(fourier_ser.)│         │(freq_analys.)│         │  (metrics)   │
    └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    ▼
                     ┌──────────────────────────────┐
                     │   Plotly Interactive Charts  │
                     │          (plotting)          │
                     └──────────────┬───────────────┘
                                    ▼
                     ┌──────────────────────────────┐
                     │ Streamlit Multipage Frontend │
                     │  (App + Pages 1 through 6)   │
                     └──────────────────────────────┘
    ```
    """)

    st.markdown("""
    <div class="eng-card">
        <h4 style="color: #00f0ff; margin-top: 0;">📦 Key Software Modules</h4>
        <ul style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.8;">
            <li><code>src/fourier_series.py</code>: Exact analytical Fourier coefficient generators, trigonometric partial sum evaluation, and Gibbs peak detection.</li>
            <li><code>src/signal_generator.py</code>: Generates continuous analytical ideal reference waveforms strictly independent of Fourier reconstructions.</li>
            <li><code>src/frequency_analysis.py</code>: Real-valued single-sided FFT routines, windowing (Hann/Hamming/Blackman), and automated harmonic peak detection.</li>
            <li><code>src/metrics.py</code>: Mathematically sound RMSE, MAE, Max Error, Signal-to-Error Ratio (SER dB), and Parseval energy share calculations.</li>
            <li><code>src/plotting.py</code>: Plotly engineering dark theme charts, animations, harmonic isolation, and spectral diagrams.</li>
            <li><code>src/exports.py</code>: CSV serialization and academic report text generation.</li>
            <li><code>src/ui_theme.py</code>: Design tokens, CSS injection, header banners, and KPI tiles.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
