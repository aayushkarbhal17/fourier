# FourierLab — Interactive Fourier Series Visualizer
### Course Mini-Project in Signals and Systems (ECE / EE)

![Python Version](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-38BDF8.svg)
![Tests](https://img.shields.io/badge/pytest-Passing%20(17%2F17)-10B981.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

---

## 1. Project Overview & Abstract

### Academic Abstract
Fourier analysis is a cornerstone of modern electrical engineering, telecommunications, audio engineering, and linear time-invariant (LTI) systems theory. The trigonometric Fourier series demonstrates that any physically realizable periodic signal can be decomposed into an infinite sum of harmonically related sinusoids. 

**FourierLab** is a standalone, interactive, college-level Signals and Systems mini-project application built with Python, Streamlit, Plotly, NumPy, and SciPy. It provides an intuitive, high-performance visual dashboard where engineering students can explore how periodic waveforms (square, sawtooth, triangular, sine, and pulse trains) are reconstructed from their Fourier coefficients.

The application strictly separates ideal analytical continuous reference waveforms from the computed Fourier partial sums, enabling exact quantitative validation of **Parseval's energy theorem**, **root-mean-square error (RMSE)** convergence rates, **harmonic decay slopes ($1/n$ vs $1/n^2$)**, and the **Wilbraham-Gibbs phenomenon** near jump discontinuities.

---

## 2. Problem Statement & Motivation

In traditional university engineering courses, Fourier series is predominantly taught as a pencil-and-paper exercise involving manual definite integrals. While mathematically sound, this approach introduces common pedagogical difficulties:
1. **Lack of Dynamic Intuition:** Students cannot visualize how adding successive high-frequency harmonics cancels out ripples and sharpens transition slopes.
2. **Misunderstanding Gibbs Phenomenon:** Textbooks note that the series converges to the midpoint average at jump discontinuities, but students rarely observe that the ~8.95% overshoot persists asymptotically for any finite number of harmonics $N$.
3. **Disconnection Between Domains:** Fourier coefficients $a_n$ and $b_n$ are often viewed as abstract algebra rather than physical spectral lines observable on a spectrum analyzer or oscilloscope.
4. **Tooling Barriers:** Proprietary tools like MATLAB require heavy local installations and expensive institutional licenses. FourierLab provides a lightweight, 100% free, local Python-native alternative that runs smoothly on standard student laptops.

---

## 3. Project Objectives

- **Analytical Rigor:** Implement exact mathematical Fourier series formulas and integrals across multiple canonical periodic waveforms.
- **Dynamic Visualization:** Build interactive Plotly charts featuring zoom, pan, hover coordinate inspection, and progressive harmonic synthesis animation.
- **Quantitative Fidelity:** Compute real-time error metrics (RMSE, MAE, Max Error, Signal-to-Error Ratio, Parseval Energy Share).
- **Spectral Exploration:** Provide single-sided Fast Fourier Transform (FFT) amplitude spectra with automated peak detection and windowing comparisons.
- **Comprehensive Academic Dossier:** Include a full textbook curriculum (16 topics), step-by-step worked derivations, 5 guided laboratory experiments, a 16-question self-assessment quiz, and a 16-question oral viva examination guide.

---

## 4. Key Features

- **Multi-Waveform Simulation:** Square waves (odd symmetry), sawtooth ramps (all integer harmonics), triangular waves (fast $1/n^2$ convergence), pure sinusoids, and rectangular pulse trains with adjustable duty cycle ($d$).
- **Flexible Parameters:** Harmonics from $N=1$ to $100$, fundamental frequency $f_0$ from $0.1$ to $20.0$ Hz, peak amplitude $A$, phase offset $\phi$, DC bias, and sampling resolution up to 3,000 points/period.
- **Gibbs Phenomenon Zoom:** Dedicated close-up analysis around discontinuities displaying empirical peak overshoot alongside the theoretical Wilbraham-Gibbs limit (~17.9% of amplitude $A$).
- **Harmonic Explorer:** Isolates individual harmonic components $h_n(t)$, displays magnitude $|c_n|$ and phase $\theta_n$, plots cumulative partial sums, and generates downloadable coefficient tables.
- **Frequency Analysis & FFT:** Computes single-sided FFT spectra scaled mathematically to true peak amplitudes, identifying harmonic peaks with diamond markers.
- **Interactive Quiz:** 16 multiple-choice questions with instant scoring, immediate explanations, progress tracking, and restart options.
- **Viva Preparation:** 16 external examiner questions with model technical answers.
- **Data Exporting:** Download time-domain waveforms, harmonic coefficient tables, and frequency spectra as clean CSV files, or generate formatted simulation text reports with custom student submission details.

---

## 5. Mathematical Foundation

### 5.1 The Trigonometric Fourier Series
For any periodic signal $x(t)$ with fundamental period $T$ and fundamental angular frequency:
$$\omega_0 = \frac{2\pi}{T} = 2\pi f_0$$

The trigonometric Fourier series partial sum $x_N(t)$ truncated to $N$ harmonics is given by:
$$x_N(t) = \frac{a_0}{2} + \sum_{n=1}^{N} \left[ a_n \cos(n\omega_0 t) + b_n \sin(n\omega_0 t) \right]$$

In polar amplitude-phase form:
$$x_N(t) = c_0 + \sum_{n=1}^{N} c_n \cos(n\omega_0 t - \theta_n)$$
where $c_0 = \frac{a_0}{2}$, $c_n = \sqrt{a_n^2 + b_n^2}$, and $\theta_n = \operatorname{atan2}(b_n, a_n)$.

### 5.2 Coefficient Formulas
Using the mutual orthogonality of sinusoidal basis functions over any period interval $[t_0, t_0 + T]$:

$$a_0 = \frac{2}{T} \int_{0}^{T} x(t) \, dt \quad \left(\text{giving the DC mean } \frac{a_0}{2} = \frac{1}{T} \int_{0}^{T} x(t) \, dt \right)$$

$$a_n = \frac{2}{T} \int_{0}^{T} x(t) \cos(n\omega_0 t) \, dt \quad (n = 1, 2, 3, \dots)$$

$$b_n = \frac{2}{T} \int_{0}^{T} x(t) \sin(n\omega_0 t) \, dt \quad (n = 1, 2, 3, \dots)$$

### 5.3 Waveform Coefficients Implemented in FourierLab

| Waveform | Symmetry Characteristics | DC Offset ($a_0/2$) | Cosine Coeffs ($a_n$) | Sine Coeffs ($b_n$) | Decay Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Square Wave** | Half-wave odd symmetry | $0$ | $0$ (for all $n$) | $\frac{4A}{n\pi}$ for odd $n$; $0$ for even $n$ | $1/n$ ($-20$ dB/dec) |
| **Sawtooth Wave** | Centered odd ramp | $0$ | $0$ (for all $n$) | $\frac{2A}{n\pi}(-1)^{n+1}$ for all $n \ge 1$ | $1/n$ ($-20$ dB/dec) |
| **Triangular Wave** | Continuous odd symmetry | $0$ | $0$ (for all $n$) | $\frac{8A}{\pi^2 n^2}(-1)^{(n-1)/2}$ for odd $n$; $0$ even | $1/n^2$ ($-40$ dB/dec) |
| **Sine Wave** | Pure fundamental sinusoid | $0$ | $0$ (for all $n$) | $b_1 = A$; $b_n = 0$ for $n \ge 2$ | Single tone |
| **Pulse Train** | Unipolar rectangular pulse | $A \cdot d$ | $\frac{A}{n\pi}\sin(2\pi n d)$ | $\frac{A}{n\pi}[1 - \cos(2\pi n d)]$ | Sinc envelope |

---

## 6. Technology Stack

- **Language:** Python 3.11+
- **Numerical Processing:** NumPy ($\ge 1.26$) for vectorized array operations and partial sum evaluation
- **Signal Processing:** SciPy ($\ge 1.11$) for discrete transforms and peak detection utilities
- **Web UI & Dashboard:** Streamlit ($\ge 1.35$) with custom multipage navigation and glassmorphism dark navy design
- **Interactive Charting:** Plotly ($\ge 5.20$) for GPU-accelerated interactive web charts and frame animations
- **Tabular Data:** Pandas ($\ge 2.0$) for coefficient tables and CSV formatting
- **Test Suite:** pytest ($\ge 8.0$) for automated numerical and analytical validation

---

## 7. Project Directory Structure

```text
FourierLab/
├── app.py                         # Main Entry Point & Dashboard
├── pages/
│   ├── 1_Signal_Lab.py            # Main Fourier Series Simulator & Gibbs Analyzer
│   ├── 2_Harmonic_Explorer.py     # Single Harmonic Isolation & Coefficient Table
│   ├── 3_Frequency_Analysis.py    # Single-Sided FFT Spectrum & Peak Detection
│   ├── 4_Theory_and_Learning.py   # 16 Textbook Topics, Derivations & 5 Guided Labs
│   ├── 5_Quiz_and_Viva.py         # 16 Interactive MCQs & 16 Viva Q&As
│   └── 6_About_Project.py         # Academic Abstract & Student Submission Dossier
├── src/
│   ├── __init__.py                # Package initialization
│   ├── fourier_series.py          # Exact mathematical coefficients & partial sum engine
│   ├── signal_generator.py        # Independent continuous ideal reference generator
│   ├── frequency_analysis.py      # Real-valued single-sided FFT, windowing & peak finder
│   ├── metrics.py                 # RMSE, MAE, Max Error, SER & Parseval energy metrics
│   ├── plotting.py                # Plotly interactive themes, animations & stem plots
│   ├── exports.py                 # CSV serialization and simulation text reports
│   └── ui_theme.py                # Design tokens, CSS injection, KPI tiles & headers
├── tests/
│   ├── test_fourier_series.py     # Analytical tests for coefficients and partial sums
│   ├── test_frequency_analysis.py # FFT normalization, peaks, and windowing tests
│   └── test_metrics.py            # Error metrics and Parseval energy validation tests
├── assets/
│   └── README.md                  # Static asset documentation
├── .streamlit/
│   └── config.toml                # Dark navy & electric cyan theme configuration
├── requirements.txt               # Pinned project dependencies
├── README.md                      # Comprehensive user & developer manual
├── .gitignore                     # Git ignore rules for Python artifacts
└── LICENSE                        # MIT License
```

---

## 8. Prerequisites

- Python 3.11, 3.12, 3.13, or 3.14 installed on your computer.
- `pip` (Python package installer).
- A modern web browser (Google Chrome, Mozilla Firefox, Microsoft Edge, or Safari).
- *(Optional)* `git` for repository version control.

---

## 9. Installation & Execution Guide

### Option A: Running with Git (Linux / macOS Bash)

```bash
# 1. Clone the repository (replace with your repository URL)
git clone <YOUR_REPOSITORY_URL>
cd FourierLab

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run automated test suite
pytest -v

# 6. Launch the interactive application
streamlit run app.py
```

### Option B: Running with Git (Windows PowerShell)

```powershell
# 1. Clone the repository (replace with your repository URL)
git clone <YOUR_REPOSITORY_URL>
cd FourierLab

# 2. Create a virtual environment
py -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Run automated test suite
pytest -v

# 6. Launch the interactive application
streamlit run app.py
```

### Option C: Running Without Git (ZIP Download)

If you do not have Git installed or downloaded the project as a `.zip` archive:
1. Extract the `.zip` archive to a folder on your computer (e.g. `C:\Projects\FourierLab` or `~/FourierLab`).
2. Open a terminal or PowerShell prompt in that extracted directory.
3. Follow steps 2 through 6 above.

### Deactivating the Virtual Environment
When you are finished using the application, run:
```bash
deactivate
```

---

## 10. Expected Application URL

Once launched, Streamlit will display output in your terminal similar to:
```text
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open your browser and navigate to `http://localhost:8501`.

---

## 11. Troubleshooting Common Errors

### 1. PowerShell Script Execution Restricted
**Symptom:** `.\.venv\Scripts\Activate.ps1 cannot be loaded because running scripts is disabled on this system.`  
**Solution:** Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Alternatively, activate via Command Prompt (`cmd`):
```cmd
.\.venv\Scripts\activate.bat
```

### 2. Port 8501 Already in Use
**Symptom:** Port `8501` is occupied by another process.  
**Solution:** Launch Streamlit on a custom port:
```bash
streamlit run app.py --server.port 8502
```

### 3. ModuleNotFoundError
**Symptom:** `ModuleNotFoundError: No module named 'plotly'` or `'src'`  
**Solution:** Ensure your virtual environment is active (your prompt should show `(.venv)`), and reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## 12. How to Use Each Section

1. **Dashboard (`app.py`):** Executive overview with KPI badges and a live Quick-Start simulation widget.
2. **Signal Lab (`pages/1_Signal_Lab.py`):** Primary simulator. Select waveforms, adjust harmonic count $N$ up to 100, observe the time-domain waveform, inspect the Gibbs overshoot zoom box, view convergence curves, and play the synthesis animation.
3. **Harmonic Explorer (`pages/2_Harmonic_Explorer.py`):** Isolate any single harmonic $n$, inspect its magnitude $|c_n|$, sine/cosine components, phase, and % power share, view stem plots of $|c_n|$ vs $n$, and export coefficient tables to CSV.
4. **Frequency Analysis (`pages/3_Frequency_Analysis.py`):** Compare the ideal continuous signal spectrum with the reconstructed signal spectrum via a single-sided FFT. Inspect detected harmonic peaks and examine the $1/n$ vs $1/n^2$ decay behavior.
5. **Theory & Learning (`pages/4_Theory_and_Learning.py`):** 16 textbook topics with LaTeX formulas, step-by-step integral derivation of a square wave, and 5 interactive guided experiments.
6. **Quiz & Viva (`pages/5_Quiz_and_Viva.py`):** 16 interactive MCQs with instant scoring and explanations, plus 16 viva questions with model examiner answers.
7. **About Project (`pages/6_About_Project.py`):** Enter student details (Name, Roll No., Department, College) to generate a customized academic submission dossier.

---

## 13. Reproducing the Educational Experiments

- **Experiment A (Effect of Harmonics):** Navigate to *Theory & Learning* $\to$ *Guided Experiments* $\to$ *Experiment A*. Increase $N$ from 1 to 40 and observe RMSE dropping monotonically toward zero.
- **Experiment B (Gibbs Phenomenon):** Select *Experiment B*. Observe the square wave close-up near $t = 0.5$ s. Notice that while the ripples compress as $N$ increases, the peak overshoot remains at ~17.9% above amplitude $A$.
- **Experiment C (Frequency Variation):** Select *Experiment C*. Adjust $f_0$ from 1 to 10 Hz and confirm that period $T = 1/f_0$ shrinks while harmonic peaks shift to integer multiples $n f_0$.
- **Experiment D (Waveform Comparison):** Select *Experiment D*. Compare square wave ($1/n$ decay) vs triangular wave ($1/n^2$ decay) at $N=5$. Observe that triangle wave captures >99.8% of signal power in 5 harmonics.
- **Experiment E (Time-Frequency Duality):** Select *Experiment E*. Observe how truncated harmonics in the time domain correspond directly to isolated spectral lines in the FFT.

---

## 14. Automated Testing

FourierLab includes 17 automated tests written for `pytest` covering:
- Exact analytical square-wave coefficients.
- Exact analytical sawtooth-wave coefficients.
- Exact analytical triangular-wave coefficients.
- Pure sine and pulse train coefficient verification.
- Output array shapes and numerical finiteness.
- Gibbs overshoot calculation.
- Harmonic table construction and zero-flagging.
- Single-sided FFT amplitude normalization and peak detection.
- DC offset detection in FFT.
- Windowing functions (Rectangular, Hann, Hamming, Blackman).
- RMSE, MAE, Max Error, and Signal-to-Error Ratio calculations.
- Parseval energy convergence.
- Convergence data generation.
- CSV export structure and summary report generation.

To run the full test suite:
```bash
pytest -v
```

Expected output:
```text
============================= test session starts =============================
platform ... pytest-9.1.1
collected 17 items

tests/test_fourier_series.py::test_square_coefficients_analytical PASSED [  5%]
tests/test_fourier_series.py::test_sawtooth_coefficients_analytical PASSED [ 11%]
tests/test_fourier_series.py::test_triangle_coefficients_analytical PASSED [ 17%]
tests/test_fourier_series.py::test_sine_coefficients PASSED              [ 23%]
tests/test_fourier_series.py::test_pulse_train_coefficients PASSED       [ 29%]
tests/test_fourier_series.py::test_fourier_series_shape_and_finite PASSED [ 35%]
tests/test_fourier_series.py::test_gibbs_overshoot_square PASSED         [ 41%]
tests/test_fourier_series.py::test_harmonic_table_generation PASSED      [ 47%]
tests/test_frequency_analysis.py::test_fft_pure_sine_amplitude_and_frequency PASSED [ 52%]
tests/test_frequency_analysis.py::test_fft_dc_offset_amplitude PASSED    [ 58%]
tests/test_frequency_analysis.py::test_find_spectral_peaks_square_wave PASSED [ 64%]
tests/test_frequency_analysis.py::test_fft_windowing_types PASSED        [ 70%]
tests/test_metrics.py::test_error_metrics_identical_signals PASSED       [ 76%]
tests/test_metrics.py::test_error_metrics_known_difference PASSED        [ 82%]
tests/test_metrics.py::test_parseval_power_share_convergence PASSED      [ 88%]
tests/test_metrics.py::test_compute_convergence_data PASSED              [ 94%]
tests/test_metrics.py::test_exports_csv_and_report PASSED                [100%]

============================= 17 passed in 0.51s ==============================
```

---

## 15. Initializing a Git Repository & Pushing to GitHub

To store this project in your own GitHub account:

```bash
# 1. Initialize git repository
git init

# 2. Stage all project files
git add .

# 3. Create initial commit
git commit -m "Initial commit: FourierLab complete Signals and Systems mini project"

# 4. Set default branch to main
git branch -M main

# 5. Link to your remote repository (replace with your actual GitHub URL)
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# 6. Push to GitHub
git push -u origin main
```

---

## 16. Academic Mini-Project Submission Dossier

Students presenting this project for university laboratory evaluation can print the **Student Submission Dossier** generated in Section 6 (`pages/6_About_Project.py`). 

### Checklist for Viva Voce Demonstration:
- [x] Explain the distinction between continuous time $t$ and sampled numerical arrays.
- [x] Demonstrate how only odd harmonics exist for square and triangular waves due to half-wave symmetry.
- [x] Zoom into the square wave transition to demonstrate the 8.95% Gibbs overshoot.
- [x] Compare the $1/n$ decay of square waves with the $1/n^2$ decay of triangular waves.
- [x] Show Parseval's energy share approaching 100% as $N$ increases.
- [x] Download the simulation report and CSV datasets to show quantitative data collection.

---

## 17. License

This project is licensed under the **MIT License** — see the [LICENSE](file:///c:/Users/Asus/fourier/LICENSE) file for complete details.
