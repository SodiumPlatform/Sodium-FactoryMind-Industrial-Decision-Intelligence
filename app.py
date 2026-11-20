"""FactoryMind - AI-Powered Industrial Decision Intelligence.

Entry point. Run with:
    streamlit run app.py

Multi-page routing is handled here (sidebar radio -> page module render()),
which gives full control over branding and keeps the app a single Streamlit
process with no external services.

Decision-support prototype: predictions and recommendations are advisory and
require human approval. Not an autonomous industrial control system.

Team Sodium | International AI Builders Congress (hackathon MVP).
"""
from __future__ import annotations

import importlib

import streamlit as st

from src import config
from src.visualization import pio  # ensures the factorymind template is registered

CUSTOM_CSS = """
<style>
/* page shell */
.block-container {padding-top: 1.35rem; padding-bottom: 2.2rem; max-width: 1180px;}
#MainMenu, footer, header {visibility: hidden;}

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A101F 0%, #0D1730 100%);
    border-right: 1px solid #1C2A44;
}
.fm-brand {display:flex; align-items:center; gap:11px; padding:6px 2px 14px 2px;}
.fm-logo {width:42px; height:42px; border-radius:11px; flex-shrink:0;
    background:linear-gradient(135deg, #3B82F6 0%, #22D3EE 100%);
    display:flex; align-items:center; justify-content:center;
    font-weight:800; color:#06121F; font-size:15px; letter-spacing:0.5px;}
.fm-name {font-weight:800; color:#F1F5F9; font-size:17px; line-height:1.1;}
.fm-tagline {color:#7DD3FC; font-size:10px; letter-spacing:0.6px; text-transform:uppercase;}

/* sidebar nav pills */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    width:100%; border-radius:9px; padding:7px 10px;
    border:1px solid transparent; margin-bottom:2px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:#12203A; border-color:#1E2A44;
}

/* metric cards */
div[data-testid="stMetric"] {
    background:#101B31; border:1px solid #1E2A44; border-radius:12px;
    padding:14px 16px 10px 16px;
}
div[data-testid="stMetricLabel"] p {color:#93A4BC !important; font-size:0.86rem;}

/* generic surfaces */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stExpander"] {
    background:#0E1728; border:1px solid #1E2A44; border-radius:12px;
}
div[data-testid="stTabs"] button {color:#93A4BC; font-weight:600;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#22D3EE;}

/* badges */
.fm-badge {display:inline-block; padding:3px 10px; border-radius:999px;
    font-size:12px; font-weight:700; letter-spacing:0.2px;}
.fm-badge.high {background:rgba(248,113,113,0.14); color:#F87171; border:1px solid rgba(248,113,113,0.45);}
.fm-badge.medium {background:rgba(251,191,36,0.14); color:#FBBF24; border:1px solid rgba(251,191,36,0.45);}
.fm-badge.low {background:rgba(52,211,153,0.14); color:#34D399; border:1px solid rgba(52,211,153,0.45);}
.fm-badge.info {background:rgba(34,211,238,0.12); color:#22D3EE; border:1px solid rgba(34,211,238,0.4);}
.fm-badge.muted {background:rgba(147,164,188,0.12); color:#93A4BC; border:1px solid rgba(147,164,188,0.35);}

/* markdown headings */
h1, h2, h3 {letter-spacing:-0.2px; color:#E9EFFA;}
hr {border-color:#1E2A44;}
</style>
"""


def _apply_page_config() -> None:
    st.set_page_config(
        page_title=f"{config.APP_NAME} - Industrial Decision Intelligence",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _brand_block() -> None:
    st.markdown(
        f"""
        <div class="fm-brand">
            <div class="fm-logo">FM</div>
            <div>
                <div class="fm-name">{config.APP_NAME}</div>
                <div class="fm-tagline">{config.APP_TAGLINE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _apply_page_config()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    with st.sidebar:
        _brand_block()

    nav_pages = [
        st.Page(lambda m=importlib.import_module(mod): m.render(),
                title=title, url_path=slug, default=default)
        for title, slug, mod, default in [
            ("Executive Dashboard", "dashboard", "pages.dashboard", True),
            ("Machine Data Upload", "data-upload", "pages.data_upload", False),
            ("AI Failure Prediction", "predictions", "pages.predictions", False),
            ("Explainable AI", "explainability", "pages.explainability", False),
            ("Maintenance Advisor", "maintenance", "pages.maintenance", False),
            ("Model Performance", "model-performance", "pages.model_performance", False),
            ("About FactoryMind", "about", "pages.about", False),
        ]
    ]
    page = st.navigation(nav_pages, position="sidebar")

    with st.sidebar:
        st.divider()
        st.caption(f"v{config.APP_VERSION} - hackathon MVP by Team Sodium")
        st.caption("Decision-support prototype. Advisory outputs only; "
                   "not an autonomous control system.")

    page.run()


if __name__ == "__main__":
    main()
