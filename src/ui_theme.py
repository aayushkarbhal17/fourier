"""
UI Theme and Styling Utilities for FourierLab
Provides a dark navy / electric cyan engineering dashboard design system.
"""

import streamlit as st


def apply_custom_css():
    """
    Applies custom CSS styling for a modern scientific engineering dashboard look.
    """
    custom_css = """
    <style>
        /* Import clean modern font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Base app styling */
        .stApp {
            background-color: #0b0f19;
            color: #f1f5f9;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Top header bar */
        header[data-testid="stHeader"] {
            background: rgba(11, 15, 25, 0.85);
            backdrop-filter: blur(12px);
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0e1526;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }

        /* Gradient titles */
        .gradient-title {
            background: linear-gradient(135deg, #00f0ff 0%, #38bdf8 50%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .gradient-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }

        /* Engineering Card Container */
        .eng-card {
            background: rgba(19, 27, 46, 0.7);
            border: 1px solid rgba(56, 189, 248, 0.15);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(8px);
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3);
            transition: border-color 0.2s ease, transform 0.2s ease;
        }

        .eng-card:hover {
            border-color: rgba(0, 240, 255, 0.35);
        }

        /* Metric Tile */
        .kpi-container {
            display: flex;
            flex-direction: column;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 1rem;
            border-left: 3px solid #00f0ff;
            margin-bottom: 0.75rem;
        }

        .kpi-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }

        .kpi-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #f8fafc;
            font-family: 'JetBrains Mono', monospace;
        }

        .kpi-sub {
            font-size: 0.75rem;
            color: #38bdf8;
            margin-top: 0.25rem;
        }

        /* Pill / Badge */
        .eng-badge {
            display: inline-block;
            background: rgba(0, 240, 255, 0.12);
            color: #00f0ff;
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 9999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.75rem;
        }

        .eng-badge-purple {
            background: rgba(168, 85, 247, 0.12);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }

        .eng-badge-green {
            background: rgba(16, 185, 129, 0.12);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        /* Code snippets */
        code {
            background-color: rgba(15, 23, 42, 0.9) !important;
            color: #38bdf8 !important;
            font-family: 'JetBrains Mono', monospace !important;
            padding: 0.15rem 0.4rem !important;
            border-radius: 4px !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.15), rgba(168, 85, 247, 0.15));
            border: 1px solid rgba(0, 240, 255, 0.4);
            color: #f8fafc;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(0, 240, 255, 0.3), rgba(168, 85, 247, 0.3));
            border-color: #00f0ff;
            color: #ffffff;
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.3);
            transform: translateY(-1px);
        }

        /* Radio and selectbox styling */
        div[data-baseweb="select"] {
            border-radius: 8px;
        }

        /* Callout boxes */
        .callout-info {
            background: rgba(14, 165, 233, 0.1);
            border-left: 4px solid #0284c7;
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            color: #e0f2fe;
        }

        .callout-warning {
            background: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            color: #fef3c7;
        }

        .callout-success {
            background: rgba(16, 185, 129, 0.1);
            border-left: 4px solid #10b981;
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            color: #d1fae5;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge_text: str = "Signals & Systems Mini Project", badge_color: str = "cyan"):
    """
    Renders standardized hero header banner for all pages.
    """
    badge_class = "eng-badge"
    if badge_color == "purple":
        badge_class = "eng-badge eng-badge-purple"
    elif badge_color == "green":
        badge_class = "eng-badge eng-badge-green"

    html = f"""
    <div style="margin-bottom: 1.5rem;">
        <span class="{badge_class}">{badge_text}</span>
        <h1 class="gradient-title">{title}</h1>
        <p class="gradient-subtitle">{subtitle}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_kpi(label: str, value: str, sub: str = "", border_color: str = "#00f0ff"):
    """
    Renders an engineering KPI metric tile with custom styling.
    """
    html = f"""
    <div class="kpi-container" style="border-left-color: {border_color};">
        <div class="kpi-title">{label}</div>
        <div class="kpi-value">{value}</div>
        {f'<div class="kpi-sub">{sub}</div>' if sub else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
