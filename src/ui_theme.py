"""
UI/UX Theme, Glassmorphic Styling & Plotly Layout Templates
Urban Atmospheric Telemetry & Machine Learning Command Center
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

THEME_COLORS = {
    'bg_primary': '#0B0F19',
    'bg_card': '#111827',
    'bg_card_trans': 'rgba(17, 24, 39, 0.75)',
    'border_subtle': 'rgba(255, 255, 255, 0.08)',
    'cyan': '#00F2FE',
    'cyan_glow': 'rgba(0, 242, 254, 0.25)',
    'emerald': '#10B981',
    'emerald_glow': 'rgba(16, 185, 129, 0.25)',
    'violet': '#8B5CF6',
    'violet_glow': 'rgba(139, 92, 246, 0.25)',
    'amber': '#F59E0B',
    'coral': '#EF4444',
    'text_primary': '#F8FAFC',
    'text_secondary': '#94A3B8',
    'text_muted': '#64748B'
}

CUSTOM_CSS = """
<style>
/* Import Monospace & Modern Sans Fonts */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

/* Base Application Background & Fonts */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #F8FAFC;
    background-color: #0B0F19 !important;
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(14, 165, 233, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(139, 92, 246, 0.07) 0%, transparent 45%),
                #0B0F19 !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0d1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0.7rem;
}

/* Glassmorphic Container Cards */
.telemetry-card {
    background: rgba(17, 24, 39, 0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

.telemetry-card:hover {
    border-color: rgba(0, 242, 254, 0.35);
    box-shadow: 0 10px 30px rgba(0, 242, 254, 0.08);
    transform: translateY(-2px);
}

.card-cyan {
    border-left: 3px solid #00F2FE;
}

.card-emerald {
    border-left: 3px solid #10B981;
}

.card-violet {
    border-left: 3px solid #8B5CF6;
}

.card-amber {
    border-left: 3px solid #F59E0B;
}

.card-coral {
    border-left: 3px solid #EF4444;
}

/* Header & Telemetry Badges */
.telemetry-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.3rem 0.65rem;
    border-radius: 9999px;
    background: rgba(0, 242, 254, 0.1);
    color: #00F2FE;
    border: 1px solid rgba(0, 242, 254, 0.25);
    margin-bottom: 0.5rem;
}

.pulse-dot {
    width: 7px;
    height: 7px;
    background-color: #00F2FE;
    border-radius: 50%;
    box-shadow: 0 0 8px #00F2FE;
    animation: pulseGlow 1.8s infinite;
}

@keyframes pulseGlow {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 254, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 6px rgba(0, 242, 254, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 242, 254, 0); }
}

/* Metric Display Numbers */
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #F8FAFC;
    line-height: 1.1;
    margin: 0.25rem 0 0.15rem 0;
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
}

.metric-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748B;
}

/* Top App Title Banner */
.app-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 30%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    font-size: 0.95rem;
    color: #94A3B8;
    margin-bottom: 1.5rem;
}

/* Button & Selectbox Refinements */
.stButton > button {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
}

.stButton > button:hover {
    border-color: #00F2FE !important;
    color: #00F2FE !important;
    box-shadow: 0 4px 16px rgba(0, 242, 254, 0.25) !important;
    transform: translateY(-1px);
}

/* Dataframe & Table Dark Mode Customization */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    overflow: hidden;
}

/* Status Callouts */
.callout-box {
    background: rgba(30, 41, 59, 0.6);
    border-left: 4px solid #00F2FE;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #CBD5E1;
}

/* Radio & Navigation Pills */
div[role="radiogroup"] label {
    background: transparent !important;
    border: none !important;
}
</style>
"""


def apply_custom_css():
    """Injects the cyber-dark CSS theme into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_plotly_layout(
    title: str = "",
    height: int = 420,
    showlegend: bool = True
) -> dict:
    """
    Returns a unified Plotly dark theme layout matching the telemetry command center design.
    """
    return dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family='Plus Jakarta Sans, sans-serif', size=15, color='#F8FAFC'),
            x=0.01,
            y=0.96
        ),
        paper_bgcolor='rgba(11, 15, 25, 0)',
        plot_bgcolor='rgba(17, 24, 39, 0.45)',
        font=dict(
            family="'JetBrains Mono', 'Plus Jakarta Sans', monospace",
            size=11,
            color='#94A3B8'
        ),
        margin=dict(l=45, r=25, t=55 if title else 25, b=45),
        height=height,
        showlegend=showlegend,
        legend=dict(
            bgcolor='rgba(17, 24, 39, 0.75)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1,
            font=dict(color='#CBD5E1', size=10.5)
        ),
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.06)',
            zerolinecolor='rgba(255, 255, 255, 0.14)',
            tickfont=dict(color='#94A3B8', size=10),
            title_font=dict(color='#E2E8F0', size=11)
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.06)',
            zerolinecolor='rgba(255, 255, 255, 0.14)',
            tickfont=dict(color='#94A3B8', size=10),
            title_font=dict(color='#E2E8F0', size=11)
        ),
        hoverlabel=dict(
            bgcolor='#111827',
            bordercolor='#00F2FE',
            font=dict(family='JetBrains Mono', size=12, color='#F8FAFC')
        )
    )


def apply_telemetry_theme(fig: go.Figure, title: str = "", height: int = 420, showlegend: bool = True) -> go.Figure:
    """Updates a Plotly figure with the custom command center theme."""
    layout_dict = get_plotly_layout(title=title, height=height, showlegend=showlegend)
    fig.update_layout(**layout_dict)
    return fig
