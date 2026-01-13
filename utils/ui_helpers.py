"""
Utility functions for consistent page styling and headers
"""
import streamlit as st

def page_header(title: str, subtitle: str, icon: str = "📈"):
    """Premium gradient page header used across all pages."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-header__icon">{icon}</div>
            <div>
                <h1 class="page-header__title">{title}</h1>
                <p class="page-header__subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def premium_metric(label: str, value: str, change: str = None, is_negative: bool = False):
    """Shared metric chip with subtle gradients."""
    change_html = ""
    if change:
        color = "#EF4444" if is_negative else "#10B981"
        sign = "" if is_negative else "+"
        change_html = f'<p style="color: {color}; margin: 8px 0 0 0; font-size: 13px; font-weight: 600;">{sign}{change}</p>'
    
    st.markdown(
        f"""
        <div class="metric-chip">
            <p class="metric-chip__label">{label}</p>
            <p class="metric-chip__value">{value}</p>
            {change_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def premium_css():
    """Apply the StockEdge-inspired styling globally."""
    custom_css = """
    <style>
        :root {
            --primary: #2d9cf4;
            --primary-2: #3b82f6;
            --surface: rgba(18, 25, 46, 0.75);
            --surface-2: #141a2d;
            --surface-3: #111526;
            --border: #1f2940;
            --text: #e9f0ff;
            --muted: #99a8c5;
            --success: #6ce2a0;
            --danger: #f87171;
            --warning: #f5c26b;
            --shadow: 0 20px 60px rgba(0,0,0,0.35);
        }

        * {
            font-family: "Inter", "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
            letter-spacing: 0.01em;
        }

        body {
            background: radial-gradient(120% 80% at 20% 20%, rgba(74, 108, 247, 0.12), transparent),
                        radial-gradient(100% 70% at 80% 0%, rgba(0, 214, 255, 0.15), transparent),
                        #0b1020;
            color: var(--text);
        }

        .stApp {
            background: transparent;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1425 0%, #0b1020 100%) !important;
            border-right: 1px solid var(--border);
            box-shadow: 8px 0 40px rgba(0,0,0,0.25);
        }

        [data-testid="stSidebar"] .stButton>button {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.05);
            color: var(--text);
            height: 46px;
            border-radius: 12px;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stButton>button:hover {
            border-color: var(--primary);
            background: rgba(59,130,246,0.12);
        }

        .sidebar-title {
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 18px 0 10px 2px;
        }

        .sidebar-brand {
            padding: 10px 4px 14px;
            text-align: center;
        }
        .sidebar-brand h1 {
            font-size: 22px;
            color: var(--text);
            margin: 0;
        }
        .sidebar-brand p {
            color: var(--muted);
            margin: 4px 0 0;
            font-size: 12px;
        }

        /* Inputs & selects */
        input, textarea, select {
            background: rgba(255,255,255,0.03) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
        }
        input:focus, textarea:focus, select:focus {
            outline: none !important;
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(59,130,246,0.18) !important;
        }
        ::placeholder {
            color: var(--muted) !important;
        }

        /* Buttons */
        button[kind="primary"], .primary-btn {
            background: linear-gradient(135deg, #2d9cf4 0%, #3b6df6 100%) !important;
            border: none !important;
            color: #fff !important;
            border-radius: 12px !important;
            height: 48px;
            font-weight: 700 !important;
            box-shadow: 0 12px 30px rgba(59, 130, 246, 0.35) !important;
        }
        button[kind="primary"]:hover, .primary-btn:hover {
            transform: translateY(-1px);
            box-shadow: 0 16px 40px rgba(59, 130, 246, 0.45) !important;
        }

        .ghost-btn {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            color: var(--text);
            border-radius: 12px;
            padding: 12px 18px;
            font-weight: 600;
        }

        /* Cards */
        .glass-card {
            background: var(--surface);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }
        .glass-card--tight { padding: 14px; }

        .hero-card {
            background: linear-gradient(135deg, #1f2c48 0%, #0f172a 80%);
            border: 1px solid rgba(59,130,246,0.35);
            border-radius: 18px;
            padding: 28px;
            box-shadow: var(--shadow);
        }

        .section-title {
            color: var(--text);
            font-size: 20px;
            font-weight: 800;
            margin: 0 0 12px;
        }
        .section-subtitle {
            color: var(--muted);
            margin: 0 0 20px;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 999px;
            color: var(--muted);
            font-size: 13px;
        }

        .mini-metric {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px;
            background: rgba(255,255,255,0.04);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        .mini-metric span {
            color: var(--muted);
            font-size: 13px;
        }
        .mini-metric strong {
            color: var(--text);
        }

        .list-card .item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .list-card .item:last-child { border-bottom: none; }
        .list-card .label { color: var(--text); font-weight: 700; }
        .list-card .muted { color: var(--muted); font-size: 12px; }
        .list-card .positive { color: var(--success); font-weight: 700; }
        .list-card .negative { color: var(--danger); font-weight: 700; }

        /* Metrics */
        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 16px;
            color: var(--text);
        }
        [data-testid="metric-container"] > div:first-child {
            color: var(--muted) !important;
            font-size: 12px !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        [data-testid="metric-container"] > div:last-child {
            font-size: 24px !important;
        }

        /* Header */
        .page-header {
            display: flex;
            align-items: center;
            gap: 14px;
            background: linear-gradient(135deg, #1f2c48 0%, #0f172a 80%);
            border: 1px solid rgba(59,130,246,0.35);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 22px;
            box-shadow: var(--shadow);
        }
        .page-header__icon {
            font-size: 22px;
        }
        .page-header__title {
            margin: 0;
            font-size: 22px;
            color: var(--text);
        }
        .page-header__subtitle {
            margin: 2px 0 0;
            color: var(--muted);
            font-size: 13px;
        }

        /* Table tweaks */
        [data-testid="stTable"] table, .stDataFrame, .dataframe {
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 12px;
            overflow: hidden;
        }
        th, td {
            color: var(--text) !important;
        }

        .metric-card {
            background: linear-gradient(135deg, #1a2035 0%, #0f172a 100%);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 14px;
            padding: 18px;
            box-shadow: var(--shadow);
            transition: all 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--primary);
        }

        .metric-chip {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 12px 14px;
            border-radius: 12px;
        }
        .metric-chip__label {
            color: var(--muted);
            margin: 0 0 6px;
            font-size: 12px;
            letter-spacing: 0.05em;
        }
        .metric-chip__value {
            color: var(--text);
            margin: 0;
            font-size: 24px;
            font-weight: 800;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
