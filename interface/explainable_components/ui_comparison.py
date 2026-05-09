import streamlit as st
from explainable.state_manager.pipeline_state import get_comparison, clear_comparison, STEP_LABELS
from interface.explainable_components.utils import render_section_header

def render_comparison():
    comparison_data = get_comparison()
    if not comparison_data:
        st.info("ยังไม่มีข้อมูลการเปรียบเทียบ (ข้อมูลจะปรากฏขึ้นหากคุณกดย้อนกลับไปแก้ไขขั้นตอนก่อนหน้าแล้วทำใหม่จนเสร็จ)")
        return

    render_section_header(
        "Pipeline Comparison (Split View)",
        "เปรียบเทียบความแตกต่างแบบฝั่งซ้าย (เดิม) และฝั่งขวา (ใหม่)",
    )

    def flatten_dictionary(dictionary_data, prefix=""):
        items_dict = {}
        for key, value in dictionary_data.items():
            if isinstance(value, dict):
                items_dict.update(flatten_dictionary(value, prefix=f"{key}_"))
            else:
                items_dict[prefix + key] = value
        return items_dict

    for step_key in ["upload", "cleaning", "transformation", "ml_process"]:
        if step_key not in comparison_data:
            continue
            
        step_data = comparison_data[step_key]
        previous_config = step_data.get("prev", {})
        current_config = step_data.get("curr", {})
        
        flat_previous = flatten_dictionary(previous_config)
        flat_current = flatten_dictionary(current_config)
        all_keys_list = sorted(list(set(flat_previous.keys()) | set(flat_current.keys())))

        # GitHub Header
        st.markdown(f"""
        <div style="background:#161B22; padding:8px 16px; border:1px solid #30363D; border-radius:6px 6px 0 0; margin-top:24px; border-bottom:1px solid #30363D;">
            <span style="color:#8B949E; font-family:monospace; font-size:0.85rem;">
                pipeline / <span style="color:#C9D1D9; font-weight:600;">{step_key}.cfg</span>
                <span style="margin-left:8px; background:#21262D; padding:2px 8px; border-radius:10px; font-size:0.75rem; color:#7D8590;">
                    {STEP_LABELS.get(step_key, step_key)}
                </span>
            </span>
        </div>
        <div style="border:1px solid #30363D; border-top:none; border-radius:0 0 6px 6px; overflow:hidden; background:#0D1117;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; border-bottom:1px solid #30363D;">
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; border-right:1px solid #30363D; background:#161B22; font-weight:600;">ORIGINAL CONFIG</div>
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; background:#161B22; font-weight:600;">CURRENT CONFIG</div>
            </div>
        """, unsafe_allow_html=True)
        
        diff_rows_html = []
        for key in all_keys_list:
            value_previous = flat_previous.get(key, "—")
            value_current = flat_current.get(key, "—")
            is_changed = str(value_previous) != str(value_current)
            
            left_background  = "background:rgba(248, 81, 73, 0.12);" if is_changed else "background:transparent;"
            right_background = "background:rgba(63, 185, 80, 0.15);" if is_changed else "background:transparent;"
            
            left_text_color  = "#F85149" if is_changed else "#8B949E"
            right_text_color = "#3FB950" if is_changed else "#C9D1D9"
            
            diff_rows_html.append(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; font-family:monospace; font-size:0.82rem; border-bottom:1px solid #21262D;">
                <div style="{left_background} color:{left_text_color}; padding:6px 12px; border-right:1px solid #30363D; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"-" if is_changed else " "}</span>{key}: {value_previous}
                </div>
                <div style="{right_background} color:{right_text_color}; padding:6px 12px; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"+" if is_changed else " "}</span>{key}: {value_current}
                </div>
            </div>
            """)
        
        st.markdown("".join(diff_rows_html), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("ล้างประวัติการเปรียบเทียบ", width="stretch"):
        clear_comparison()
        st.rerun()
