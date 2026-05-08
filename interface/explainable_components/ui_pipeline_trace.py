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
STEP_ICONS_INFO = {
    "Upload":              "📂",
    "Data Cleaning":       "🧹",
    "Data Transformation": "⚙️",
    "ML Process":          "🏆",
}
PIPELINE_STEP_ORDER = ["Upload", "Data Cleaning", "Data Transformation", "ML Process"]

def render_trace():
    render_section_header(
        "บันทึกการตัดสินใจตลอด Pipeline",
        "ทุกขั้นตอนตั้งแต่ Upload ถึง ML Process — พร้อมเหตุผลว่า \"ทำไม\" ระบบจึงตัดสินใจแบบนั้น",
    )

    pipeline_log = get_log()
    if not pipeline_log:
        st.info("ยังไม่มีข้อมูล — ต้องเริ่ม pipeline ตั้งแต่ขั้นตอน Upload ใหม่")
        return

    # Step progress bar
    completed_steps = {entry.get("step") for entry in pipeline_log}
    completed_count = sum(1 for step in PIPELINE_STEP_ORDER if step in completed_steps)

    # progress bar
    progress_percentage = int(completed_count / len(PIPELINE_STEP_ORDER) * 100)
    st.markdown(
        f'<div style="background:{BORDER_COLOR};border-radius:6px;height:6px;margin-bottom:8px">'
        f'<div style="background:linear-gradient(90deg,#58a6ff,#bc8cff);width:{progress_percentage}%;'
        f'height:6px;border-radius:6px;transition:width 0.3s"></div></div>',
        unsafe_allow_html=True,
    )

    progress_columns = st.columns(len(PIPELINE_STEP_ORDER))
    for col, step in zip(progress_columns, PIPELINE_STEP_ORDER):
        color   = STEP_COLORS_INFO[step]
        icon    = STEP_ICONS_INFO[step]
        is_done    = step in completed_steps
        opacity_level = "1" if is_done else "0.28"
        check_mark   = "✓" if is_done else ""
        with col:
            st.markdown(
                f'<div style="text-align:center;opacity:{opacity_level};padding:4px 0">'
                f'<div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>'
                f'<div style="color:{color};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.03em">{step}</div>'
                f'<div style="color:{color};font-size:0.7rem;margin-top:2px">{check_mark}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Timeline cards
    for index, entry in enumerate(pipeline_log):
        step_name  = entry.get("step", "")
        log_items = entry.get("items", [])
        explanations_list = entry.get("explanations", [])
        color_hex = STEP_COLORS_INFO.get(step_name, "#8b949e")
        icon_emoji  = STEP_ICONS_INFO.get(step_name, "📋")

        header_items_list = [item for item in log_items if not item.startswith("  ")]
        detail_items_list = [item for item in log_items if item.startswith("  ")]

        # WHAT happened rows
        rows_html_content = ""
        for item in header_items_list:
            if item.endswith(":"):
                rows_html_content += (
                    f'<div style="color:{color_hex};font-size:0.85rem;font-weight:700;'
                    f'letter-spacing:0.04em;text-transform:uppercase;padding:10px 0 4px">{item}</div>'
                )
            else:
                rows_html_content += (
                    f'<div style="color:{TEXT_COLOR};font-size:0.95rem;padding:4px 0;'
                    f'border-bottom:1px solid rgba(48,54,61,0.4)">{item}</div>'
                )

        # Detail items (expandable)
        details_html_content = "".join(
            f'<div style="color:{TEXT_DIM_COLOR};font-size:0.88rem;padding:3px 0">'
            f'<code style="color:{TEXT_COLOR};background:rgba(255,255,255,0.04);'
            f'border-radius:4px;padding:1px 6px">{item.strip()}</code></div>'
            for item in detail_items_list
        )

        detail_section_html = ""
        if detail_items_list:
            detail_section_html = (
                f'<details style="margin-top:10px">'
                f'<summary style="cursor:pointer;color:{TEXT_DIM_COLOR};font-size:0.88rem;'
                f'font-weight:600;user-select:none">'
                f'ดูรายละเอียด ({len(detail_items_list)} รายการ)</summary>'
                f'<div style="padding:8px 0 0 4px">{details_html_content}</div>'
                f'</details>'
            )

        # WHY explanations
        why_section_html = ""
        if explanations_list:
            why_items_html_content = ""
            for explanation in explanations_list:
                # Warning items
                if explanation.startswith("⚠"):
                    why_items_html_content += (
                        f'<div style="color:#d29922;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(210,153,34,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{explanation}</div>'
                    )
                # Success items
                elif explanation.startswith("✓"):
                    why_items_html_content += (
                        f'<div style="color:#3fb950;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(63,185,80,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{explanation}</div>'
                    )
                # Info items
                elif explanation.startswith("ℹ"):
                    why_items_html_content += (
                        f'<div style="color:#8b949e;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(139,148,158,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{explanation}</div>'
                    )
                # Question headers (ทำไม...?)
                elif explanation.startswith("ทำไม"):
                    why_items_html_content += (
                        f'<div style="color:{TEXT_COLOR};font-size:0.92rem;padding:6px 10px;'
                        f'background:rgba(88,166,255,0.06);border-left:2px solid {color_hex};'
                        f'border-radius:0 6px 6px 0;'
                        f'margin:4px 0;line-height:1.7">{explanation}</div>'
                    )
                # Sub-items (  → ...)
                elif explanation.startswith("  →"):
                    why_items_html_content += (
                        f'<div style="color:{TEXT_DIM_COLOR};font-size:0.88rem;'
                        f'padding:2px 10px 2px 20px;line-height:1.5">{explanation}</div>'
                    )
                # General explanations
                else:
                    why_items_html_content += (
                        f'<div style="color:{TEXT_COLOR};font-size:0.9rem;padding:5px 10px;'
                        f'margin:3px 0;line-height:1.6">{explanation}</div>'
                    )

            why_section_html = (
                f'<div style="margin-top:14px;padding-top:14px;'
                f'border-top:1px dashed rgba(48,54,61,0.6)">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">'
                f'<span style="font-size:1rem">💡</span>'
                f'<span style="color:{color_hex};font-weight:700;font-size:0.88rem;'
                f'letter-spacing:0.05em;text-transform:uppercase">ทำไม?</span></div>'
                f'{why_items_html_content}'
                f'</div>'
            )

        # Assemble card
        st.markdown(
            f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
            f'border-left:3px solid {color_hex};border-radius:0 {BORDER_RADIUS} {BORDER_RADIUS} 0;'
            f'padding:{PADDING_STYLE};{MARGIN_GAP}">'
            # Header with icon + step name
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
            f'<span style="font-size:1.3rem">{icon_emoji}</span>'
            f'<span style="color:{color_hex};font-weight:700;font-size:0.95rem;'
            f'letter-spacing:0.04em;text-transform:uppercase">{step_name}</span></div>'
            # WHAT
            f'{rows_html_content}{detail_section_html}'
            # WHY
            f'{why_section_html}'
            f'</div>',
            unsafe_allow_html=True,
        )
