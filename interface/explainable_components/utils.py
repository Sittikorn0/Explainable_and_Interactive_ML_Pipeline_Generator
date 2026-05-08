import streamlit as st

# Design tokens
BACKGROUND_COLOR       = "#161b22"
BORDER_COLOR   = "#30363d"
TEXT_COLOR     = "#c9d1d9"
TEXT_DIM_COLOR = "#8b949e"
BORDER_RADIUS        = "10px"
PADDING_STYLE      = "16px 20px"
MARGIN_GAP      = "margin: 12px 0"

def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = (
        f'<div style="color:{TEXT_DIM_COLOR};font-size:0.9rem;margin-top:4px">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="{MARGIN_GAP}">'
        f'<div style="color:#e6edf3;font-size:1.15rem;font-weight:700">{title}</div>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )
