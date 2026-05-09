import streamlit as st

# Design tokens - Midnight Lab Palette (Flat Modern)
BACKGROUND_COLOR       = "#24283B"  # Deep Navy Surface
BORDER_COLOR   = "#414868"          # Sharp Slate Border
TEXT_COLOR     = "#7AA2F7"          # Soft Blue for Headings
TEXT_DIM_COLOR = "#94a3b8"          # Muted Slate
ACCENT_BLUE    = "#7AA2F7"          # Soft Blue
ACCENT_INSIGHT = "#9ECE6A"          # Lime Tea
BORDER_RADIUS        = "4px"        # Minimal Sharp Corner
PADDING_STYLE      = "1.25rem"
MARGIN_GAP      = "margin: 1.5rem 0"

def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = (
        f'<div style="color:{TEXT_DIM_COLOR};font-size:0.95rem;margin-top:4px;font-weight:400;line-height:1.6">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="{MARGIN_GAP}">'
        f'<div style="color:{TEXT_COLOR};font-size:1.3rem;font-weight:700;letter-spacing:-0.01em">{title}</div>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )
