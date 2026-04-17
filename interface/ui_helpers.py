import streamlit as st


def _badge(text: str, color: str = "blue") -> str:
    colors = {
        "blue":   ("#1a3a5c", "#58a6ff"),
        "green":  ("#1a3a2a", "#3fb950"),
        "orange": ("#3a2a10", "#d29922"),
        "red":    ("#3a1a1a", "#f85149"),
        "gray":   ("#21262d", "#8b949e"),
    }
    bg, fg = colors.get(color, colors["blue"])
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;border:1px solid {fg}33">{text}</span>'
    )


def _rec_box(title: str, reason: str, warning: str = None):
    """กล่องแสดง recommendation พร้อมเหตุผล"""
    warn_html = ""
    if warning:
        warn_html = (
            f'<div style="margin-top:8px;padding:6px 12px;background:#2d1f0a;'
            f'border-left:3px solid #d29922;border-radius:0 6px 6px 0;'
            f'font-size:0.8rem;color:#d29922">⚠ {warning}</div>'
        )
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #58a6ff;
border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0">
  <div style="font-weight:600;color:#58a6ff;font-size:0.88rem;margin-bottom:6px">
     แนะนำ: {title}
  </div>
  <div style="font-size:0.83rem;color:#c9d1d9;line-height:1.7">{reason}</div>
  {warn_html}
</div>
""", unsafe_allow_html=True)

