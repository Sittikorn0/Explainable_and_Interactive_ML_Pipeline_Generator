import streamlit as st
from explainable.state_manager.trace_log import get_log
from interface.explainable_components.utils import (
    render_section_header, BACKGROUND_COLOR, BORDER_COLOR, TEXT_COLOR, TEXT_DIM_COLOR,
    BORDER_RADIUS, PADDING_STYLE, MARGIN_GAP
)

STEP_COLORS_INFO = {
    "Upload":              "#58a6ff",
    "Data Cleaning":       "#3fb950",
    "Data Transformation": "#d29922",
    "ML Process":          "#bc8cff",
}
PIPELINE_STEP_ORDER = ["Upload", "Data Cleaning", "Data Transformation", "ML Process"]

def render_trace():
    render_section_header(
        "บันทึกการตัดสินใจตลอด Pipeline",
        "เจาะลึกทุกร่องรอยการประมวลผล — ตั้งแต่ข้อมูลดิบจนถึงโมเดลที่แม่นยำที่สุด",
    )

    pipeline_log = get_log()
    if not pipeline_log:
        st.info("ยังไม่มีข้อมูล — ระบบจะเริ่มบันทึกเมื่อคุณดำเนินการผ่านแต่ละขั้นตอน")
        return

    # CSS for Timeline
    st.markdown("""<style>
.trace-container { position: relative; padding-left: 45px; margin-top: 20px; }
.trace-line { position: absolute; left: 20px; top: 0; bottom: 0; width: 2px; background: linear-gradient(180deg, #58a6ff 0%, #30363d 100%); z-index: 0; }
.step-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 30px; position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.step-dot { position: absolute; left: -35px; top: 20px; width: 24px; height: 24px; background: #0d1117; border: 3px solid #58a6ff; border-radius: 50%; z-index: 1; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #58a6ff; font-weight: 700; }
.stat-badge { background: rgba(139, 148, 158, 0.1); border: 1px solid rgba(139, 148, 158, 0.2); border-radius: 6px; padding: 6px 12px; margin-right: 8px; margin-bottom: 8px; display: inline-block; }
.insight-bubble { background: rgba(88, 166, 255, 0.05); border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

    st.markdown('<div class="trace-container"><div class="trace-line"></div>', unsafe_allow_html=True)

    for index, entry in enumerate(pipeline_log):
        step_name = entry.get("step", "")
        log_items = entry.get("items", [])
        explanations = entry.get("explanations", [])
        color = STEP_COLORS_INFO.get(step_name, "#8b949e")

        # แยกส่วน Metrics ออกจากข้อมูลอื่นใน log_items
        metrics_html = '<div style="display: flex; flex-wrap: wrap; margin-bottom: 12px;">'
        other_items_html = ""
        
        for item in log_items:
            if ":" in item and "→" in item: # น่าจะเป็น Metric เช่น "Rows: 100 -> 80"
                label, val = item.split(":", 1)
                metrics_html += f'<div class="stat-badge"><div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase;">{label.strip()}</div><div style="font-size: 1rem; color: #e6edf3; font-weight: 600;">{val.strip()}</div></div>'
            elif not item.startswith("  "): # Header item
                other_items_html += f'<div style="font-weight: 600; color: #c9d1d9; margin: 8px 0 4px;">{item}</div>'
            else: # Detail item
                other_items_html += f'<div style="font-size: 0.85rem; color: #8b949e; padding-left: 12px; border-left: 1px solid #30363d; margin: 2px 0;">{item.strip()}</div>'
        
        metrics_html += "</div>"

        # ปรับปรุงส่วน Insights (Explanations)
        insights_html = ""
        if explanations:
            items_content = ""
            for exp in explanations:
                # ทำความสะอาดข้อความ "ทำไม..." ออกเพื่อลดความซ้ำซ้อน
                display_text = exp.replace("ทำไม", "").replace("?", "").strip()
                if exp.startswith("ทำไม"):
                    items_content += f'<div style="font-weight: 700; color: #58a6ff; font-size: 0.95rem; margin-bottom: 4px;">{display_text}</div>'
                elif exp.startswith("  →"):
                    items_content += f'<div style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; margin-bottom: 8px;">{exp.strip().replace("→", "—")}</div>'
                elif "✓" in exp or "⚠" in exp or "ℹ" in exp:
                    items_content += f'<div style="font-size: 0.88rem; color: #8b949e; margin: 4px 0;">{exp}</div>'
                else:
                    items_content += f'<div style="font-size: 0.9rem; color: #e6edf3; margin: 4px 0;">{exp}</div>'

            insights_html = f'<div style="margin-top: 20px; border-top: 1px solid #30363d; padding-top: 16px;"><div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Data Insights & Rationale</div><div class="insight-bubble">{items_content}</div></div>'

        # เรนเดอร์ Card
        st.markdown(f'<div class="step-card"><div class="step-dot" style="border-color: {color}; color: {color}">{index + 1}</div><div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;"><span style="font-size: 1.1rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 1px;">{step_name}</span></div>{metrics_html}{other_items_html}{insights_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
