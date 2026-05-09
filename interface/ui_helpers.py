import streamlit as st

SANS_FONT = "'DM Sans','Sarabun',sans-serif"


def badge(text: str, color: str = "blue") -> str:
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
        f'border-radius:12px;font-size:0.8rem;font-weight:600;border:1px solid {fg}33">{text}</span>'
    )


def recommendation_box(title: str, reason: str, warning: str = None):
    """กล่องแสดง recommendation พร้อมเหตุผล"""
    warning_html = ""
    if warning:
        warning_html = (
            f'<div style="margin-top:8px;padding:6px 12px;background:#2d1f0a;'
            f'border-left:3px solid #d29922;border-radius:0 6px 6px 0;'
            f'font-size:0.8rem;color:#d29922">{warning}</div>'
        )
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #58a6ff;
border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0">
  <div style="font-weight:600;color:#58a6ff;font-size:1rem;margin-bottom:6px">
     แนะนำ: {title}
  </div>
  <div style="font-size:1rem;color:#c9d1d9;line-height:1.7">{reason}</div>
  {warning_html}
</div>
""", unsafe_allow_html=True)

def render_metrics_row(metrics: list[tuple[str, str]]):
    """เรนเดอร์ Metrics Card แบบยืดหยุ่นด้วย HTML/CSS Flexbox ไม่ให้ข้อความโดนตัด"""
    cards_html = ""
    for label, value in metrics:
        cards_html += f"""<div style="flex: 1; min-width: 160px; background: #1E293B; border: 1px solid #334155; border-radius: 10px; padding: 16px 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);">
<div style="color: #94A3B8; font-size: 0.9rem; font-weight: 600; margin-bottom: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{label}</div>
<div style="color: #E2E8F0; font-size: 1.75rem; font-weight: 700; word-break: break-word; line-height: 1.15;">{value}</div>
</div>"""
    st.markdown(f'<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 1rem;">{cards_html}</div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    """เรนเดอร์ส่วนหัวของหน้าจอพร้อม Title และ Subtitle"""
    st.markdown(
        f'<div style="margin-bottom:2rem; margin-top:0.5rem;">'
        f'<h2 style="font-family:{SANS_FONT};font-size:1.6rem;font-weight:700;'
        f'color:#C0CAF5;margin:0 0 4px;letter-spacing:-0.02em;">{title}</h2>'
        + (
            f'<p style="font-family:{SANS_FONT};font-size:1rem;color:#787C99;margin:0;font-weight:400;">{subtitle}</p>'
            if subtitle
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )

