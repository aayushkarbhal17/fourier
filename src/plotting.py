"""
Interactive Plotly Visualization Module for FourierLab
Engineered with modern dark navy / electric cyan styling, responsive layouts,
and fast rendering performance.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Design System Color Tokens
BG_COLOR = "#0b0f19"
PANEL_BG = "#131b2e"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
TEXT_COLOR = "#f1f5f9"
MUTED_TEXT = "#94a3b8"
CYAN_ACCENT = "#00f0ff"
VIOLET_ACCENT = "#a855f7"
LIME_ACCENT = "#10b981"
AMBER_ACCENT = "#f59e0b"
ROSE_ACCENT = "#f43f5e"

CHART_THEME = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=PANEL_BG,
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", color=TEXT_COLOR, size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    xaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT_COLOR, size=13)
    ),
    yaxis=dict(
        gridcolor=GRID_COLOR,
        zerolinecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(color=MUTED_TEXT),
        title_font=dict(color=TEXT_COLOR, size=13)
    ),
    legend=dict(
        bgcolor="rgba(19, 27, 46, 0.8)",
        bordercolor=GRID_COLOR,
        borderwidth=1,
        font=dict(color=TEXT_COLOR)
    )
)


def plot_time_domain(
    t: np.ndarray,
    ideal: np.ndarray,
    reconstructed: np.ndarray,
    harmonics_matrix: Optional[np.ndarray] = None,
    show_harmonics: bool = False,
    max_harmonic_overlays: int = 4,
    title: str = "Time Domain Waveform Reconstruction"
) -> go.Figure:
    """
    Renders the primary time-domain chart comparing ideal signal, reconstructed signal,
    and optional individual harmonic components.
    """
    fig = go.Figure()

    # Individual harmonics overlays
    if show_harmonics and harmonics_matrix is not None:
        palette = [VIOLET_ACCENT, LIME_ACCENT, AMBER_ACCENT, ROSE_ACCENT, "#38bdf8"]
        n_to_show = min(max_harmonic_overlays, harmonics_matrix.shape[0])
        for i in range(n_to_show):
            # Check if this harmonic is non-trivial
            if np.max(np.abs(harmonics_matrix[i, :])) > 1e-4:
                fig.add_trace(go.Scatter(
                    x=t,
                    y=harmonics_matrix[i, :],
                    mode="lines",
                    name=f"Harmonic {i+1} (n={i+1})",
                    line=dict(color=palette[i % len(palette)], width=1.5, dash="dot"),
                    opacity=0.75,
                    hovertemplate=f"Harmonic {i+1}: %{{y:.3f}}<extra></extra>"
                ))

    # Ideal Reference Signal
    fig.add_trace(go.Scatter(
        x=t,
        y=ideal,
        mode="lines",
        name="Ideal Reference Signal",
        line=dict(color="rgba(148, 163, 184, 0.7)", width=2, dash="dash"),
        hovertemplate="t: %{x:.4f} s<br>Ideal: %{y:.3f}<extra></extra>"
    ))

    # Fourier Reconstructed Signal
    fig.add_trace(go.Scatter(
        x=t,
        y=reconstructed,
        mode="lines",
        name="Fourier Reconstruction",
        line=dict(color=CYAN_ACCENT, width=2.5),
        hovertemplate="t: %{x:.4f} s<br>Recon: %{y:.3f}<extra></extra>"
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        xaxis_title="Time t (seconds)",
        yaxis_title="Amplitude x(t)",
        hovermode="x unified",
        height=480
    )
    return fig


def plot_harmonic_explorer(
    t: np.ndarray,
    total_reconstructed: np.ndarray,
    harmonic_signal: np.ndarray,
    cumulative_signal: np.ndarray,
    n: int,
    freq_n: float,
    magnitude: float,
    phase_deg: float
) -> go.Figure:
    """
    Renders dedicated harmonic isolation chart:
    Top subplot: Selected harmonic component alone.
    Bottom subplot: Cumulative partial sum up to harmonic n vs total reconstruction.
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            f"Isolated Component #{n}: {magnitude:.4f} · sin({n}ω₀t + {phase_deg:.1f}°) at {freq_n:.2f} Hz",
            f"Cumulative Partial Sum x_{n}(t) vs Full Approximation"
        )
    )

    # Subplot 1: Isolated Harmonic
    fig.add_trace(go.Scatter(
        x=t,
        y=harmonic_signal,
        mode="lines",
        name=f"Harmonic {n}",
        line=dict(color=VIOLET_ACCENT, width=2.5),
        hovertemplate="t: %{x:.4f} s<br>h_n(t): %{y:.4f}<extra></extra>"
    ), row=1, col=1)

    # Subplot 2: Cumulative sum vs Total
    fig.add_trace(go.Scatter(
        x=t,
        y=cumulative_signal,
        mode="lines",
        name=f"Partial Sum (up to n={n})",
        line=dict(color=LIME_ACCENT, width=2),
        hovertemplate="t: %{x:.4f} s<br>Partial Sum: %{y:.3f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=t,
        y=total_reconstructed,
        mode="lines",
        name="Final Reconstruction",
        line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5, dash="dash"),
        hovertemplate="t: %{x:.4f} s<br>Final: %{y:.3f}<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(
        **CHART_THEME,
        height=520,
        hovermode="x unified",
        showlegend=True
    )
    fig.update_xaxes(title_text="Time (s)", row=2, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1, gridcolor=GRID_COLOR)

    return fig


def plot_harmonic_coefficients_stem(
    harmonic_numbers: np.ndarray,
    magnitudes: np.ndarray,
    is_zero: np.ndarray
) -> go.Figure:
    """
    Renders discrete harmonic magnitude spectrum (stem / bar chart) highlighting zero-valued harmonics.
    """
    fig = go.Figure()

    active_indices = ~is_zero
    zero_indices = is_zero

    # Active harmonics
    fig.add_trace(go.Bar(
        x=harmonic_numbers[active_indices],
        y=magnitudes[active_indices],
        name="Active Harmonics",
        marker_color=CYAN_ACCENT,
        opacity=0.85,
        hovertemplate="Harmonic n: %{x}<br>Magnitude c_n: %{y:.4f}<extra></extra>"
    ))

    # Null / Zero harmonics
    if np.any(zero_indices):
        fig.add_trace(go.Bar(
            x=harmonic_numbers[zero_indices],
            y=np.zeros(np.sum(zero_indices)),
            name="Null Harmonics (Zero)",
            marker_color="rgba(148, 163, 184, 0.4)",
            hovertemplate="Harmonic n: %{x}<br>Magnitude: 0 (Canceled by symmetry)<extra></extra>"
        ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="Harmonic Magnitude Spectrum |c_n| vs Harmonic Index (n)", font=dict(size=15)),
        xaxis_title="Harmonic Index (n)",
        yaxis_title="Harmonic Magnitude |c_n|",
        bargap=0.4,
        height=380
    )
    return fig


def plot_frequency_spectrum(
    freqs_ideal: np.ndarray,
    amps_ideal: np.ndarray,
    freqs_recon: np.ndarray,
    amps_recon: np.ndarray,
    f0: float,
    max_freq_display: float,
    peaks_df: Optional[pd.DataFrame] = None
) -> go.Figure:
    """
    Renders single-sided amplitude spectrum comparing Ideal signal FFT vs Reconstructed signal FFT.
    """
    fig = go.Figure()

    # Filter to visible range
    mask_ideal = freqs_ideal <= max_freq_display
    mask_recon = freqs_recon <= max_freq_display

    # Ideal spectrum line / area
    fig.add_trace(go.Scatter(
        x=freqs_ideal[mask_ideal],
        y=amps_ideal[mask_ideal],
        mode="lines",
        name="Ideal Signal FFT",
        line=dict(color="rgba(148, 163, 184, 0.6)", width=1.5, dash="dash"),
        hovertemplate="f: %{x:.2f} Hz<br>Amp: %{y:.4f}<extra></extra>"
    ))

    # Reconstructed spectrum
    fig.add_trace(go.Scatter(
        x=freqs_recon[mask_recon],
        y=amps_recon[mask_recon],
        mode="lines",
        name="Reconstructed Signal FFT",
        line=dict(color=CYAN_ACCENT, width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 240, 255, 0.08)",
        hovertemplate="f: %{x:.2f} Hz<br>Amp: %{y:.4f}<extra></extra>"
    ))

    # Peak markers
    if peaks_df is not None and not peaks_df.empty:
        peaks_visible = peaks_df[peaks_df["Detected Freq (Hz)"] <= max_freq_display]
        if not peaks_visible.empty:
            fig.add_trace(go.Scatter(
                x=peaks_visible["Detected Freq (Hz)"],
                y=peaks_visible["Spectral Amplitude"],
                mode="markers+text",
                name="Harmonic Peaks",
                marker=dict(color=AMBER_ACCENT, size=9, symbol="diamond"),
                text=[f"n={int(n)}" for n in peaks_visible["Harmonic (n)"]],
                textposition="top center",
                hovertemplate="Harmonic Peak n=%{text}<br>Freq: %{x:.2f} Hz<br>Amp: %{y:.4f}<extra></extra>"
            ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="Single-Sided Frequency Amplitude Spectrum |X(f)|", font=dict(size=15)),
        xaxis_title="Frequency f (Hz)",
        yaxis_title="Amplitude |X(f)|",
        height=450,
        hovermode="x unified"
    )
    fig.update_xaxes(range=[0, max_freq_display], gridcolor=GRID_COLOR)
    return fig


def plot_error_convergence(df_convergence: pd.DataFrame) -> go.Figure:
    """
    Plots RMSE, MAE, and Max Error convergence across harmonic count N.
    Demonstrates monotonic decay of RMSE/MAE and Gibbs asymptotic floor for Max Error.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_convergence["Harmonics (N)"],
        y=df_convergence["RMSE"],
        mode="lines+markers",
        name="RMSE (Root Mean Square Error)",
        line=dict(color=CYAN_ACCENT, width=2.5),
        marker=dict(size=5)
    ))

    fig.add_trace(go.Scatter(
        x=df_convergence["Harmonics (N)"],
        y=df_convergence["MAE"],
        mode="lines+markers",
        name="MAE (Mean Absolute Error)",
        line=dict(color=LIME_ACCENT, width=2),
        marker=dict(size=5)
    ))

    fig.add_trace(go.Scatter(
        x=df_convergence["Harmonics (N)"],
        y=df_convergence["Max Absolute Error"],
        mode="lines+markers",
        name="Max Absolute Error (Gibbs Peak)",
        line=dict(color=ROSE_ACCENT, width=2, dash="dash"),
        marker=dict(size=5)
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="Reconstruction Error vs Number of Harmonics (N)", font=dict(size=15)),
        xaxis_title="Number of Harmonics (N)",
        yaxis_title="Error Metric",
        height=420,
        hovermode="x unified"
    )
    return fig


def plot_gibbs_zoom(
    t: np.ndarray,
    ideal: np.ndarray,
    reconstructed: np.ndarray,
    discontinuity_time: float,
    window_sec: float
) -> go.Figure:
    """
    Zooms closely onto a jump discontinuity to demonstrate the Gibbs overshoot and ripple frequency.
    """
    mask = (t >= (discontinuity_time - window_sec)) & (t <= (discontinuity_time + window_sec))
    t_win = t[mask]
    ideal_win = ideal[mask]
    recon_win = reconstructed[mask]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_win,
        y=ideal_win,
        mode="lines",
        name="Ideal Discontinuity",
        line=dict(color="rgba(148, 163, 184, 0.7)", width=2, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=t_win,
        y=recon_win,
        mode="lines",
        name="Fourier Partial Sum (Ripples)",
        line=dict(color=ROSE_ACCENT, width=2.5)
    ))

    fig.update_layout(
        **CHART_THEME,
        title=dict(text=f"Gibbs Phenomenon: Discontinuity Close-Up (t ≈ {discontinuity_time:.3f} s)", font=dict(size=15)),
        xaxis_title="Time t (seconds)",
        yaxis_title="Amplitude",
        height=380,
        hovermode="x unified"
    )
    return fig


def create_harmonic_building_animation(
    t: np.ndarray,
    ideal: np.ndarray,
    partial_sums: List[np.ndarray],
    harmonic_steps: List[int]
) -> go.Figure:
    """
    Builds an animated Plotly figure showing progressive harmonic summation
    from n=1 up to N with play/pause slider controls.
    """
    # Initial state (first harmonic step)
    initial_y = partial_sums[0]

    fig = go.Figure(
        data=[
            go.Scatter(
                x=t,
                y=ideal,
                mode="lines",
                name="Ideal Reference",
                line=dict(color="rgba(148, 163, 184, 0.6)", width=2, dash="dash")
            ),
            go.Scatter(
                x=t,
                y=initial_y,
                mode="lines",
                name="Fourier Partial Sum",
                line=dict(color=CYAN_ACCENT, width=2.5)
            )
        ]
    )

    # Build animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter(x=t, y=ideal),
                go.Scatter(x=t, y=partial_sums[i])
            ],
            name=f"n_{harmonic_steps[i]}"
        )
        for i in range(len(partial_sums))
    ]
    fig.frames = frames

    # Slider steps
    slider_steps = [
        dict(
            args=[[f.name], dict(mode="immediate", frame=dict(duration=250, redraw=False), transition=dict(duration=0))],
            label=f"N={harmonic_steps[i]}",
            method="animate"
        )
        for i, f in enumerate(frames)
    ]

    fig.update_layout(
        **CHART_THEME,
        title=dict(text="Progressive Harmonic Synthesis Animation", font=dict(size=15)),
        xaxis_title="Time t (seconds)",
        yaxis_title="Amplitude",
        height=520,
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.0,
            y=-0.18,
            xanchor="left",
            yanchor="top",
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            font=dict(color=TEXT_COLOR),
            buttons=[
                dict(
                    label="▶ Play Synthesis",
                    method="animate",
                    args=[None, dict(frame=dict(duration=200, redraw=False), fromcurrent=True, transition=dict(duration=0))]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))]
                )
            ]
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.25,
            len=0.75,
            xanchor="left",
            y=-0.18,
            yanchor="top",
            font=dict(color=TEXT_COLOR),
            bgcolor=PANEL_BG,
            bordercolor=GRID_COLOR,
            currentvalue=dict(prefix="Synthesis Stage: ", font=dict(color=CYAN_ACCENT, size=13))
        )]
    )
    return fig
