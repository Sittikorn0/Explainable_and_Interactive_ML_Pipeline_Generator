import streamlit as st

def render_step_indicator(current_page, pages_mapping, step_status, STEP_ORDER, STEP_LABELS):
    """เรนเดอร์แถบแสดงขั้นตอนการทำงานใน Sidebar (Minimal Style)"""
    
    # 1. CSS สำหรับ Minimal List (No Boxes)
    css_rules = ""
    for index, step_key in enumerate(STEP_ORDER):
        is_active_page = (pages_mapping[step_key].url_path == current_page.url_path)
        
        # ค้นหา active index เพื่อเทียบสถานะ done
        active_idx = 0
        for i, k in enumerate(STEP_ORDER):
            if pages_mapping[k].url_path == current_page.url_path:
                active_idx = i
                break
        
        # สีตามสถานะ
        dot_color = "#414868" 
        text_color = "#565F89"
        
        if is_active_page:
            dot_color = "#7AA2F7"
            text_color = "#7AA2F7"
        elif index < active_idx:
            dot_color = "#9ECE6A"
            text_color = "#C0CAF5"
            
        css_rules += f"""
        div[class*="st-key-sidebar_step_{step_key}"] button {{
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 4px 0 !important;
            margin: 0 !important;
            min-height: unset !important;
            width: 100% !important;
            justify-content: flex-start !important;
        }}
        div[class*="st-key-sidebar_step_{step_key}"] button p::before {{
            content: '●';
            color: {dot_color};
            margin-right: 12px;
            font-size: 10px;
            vertical-align: middle;
        }}
        div[class*="st-key-sidebar_step_{step_key}"] button p {{
            color: {text_color} !important;
            font-size: 0.95rem !important;
            font-weight: {"600" if is_active_page else "400"} !important;
            white-space: nowrap !important;
        }}
        div[class*="st-key-sidebar_step_{step_key}"] button:hover p {{
            color: #FFFFFF !important;
        }}
        """

    st.sidebar.markdown(
        f"<style>\n.minimal-nav-container {{\n    padding: 1.5rem 1rem;\n}}\n{css_rules}\n</style>\n<div class=\"minimal-nav-container\">",
        unsafe_allow_html=True
    )
    
    from app import navigate
    for index, step_key in enumerate(STEP_ORDER):
        status_value = step_status[step_key]
        step_label = STEP_LABELS[step_key]
        is_active_page = (pages_mapping[step_key].url_path == current_page.url_path)

        with st.sidebar:
            button_key = f"sidebar_step_{step_key}"
            if st.button(step_label, key=button_key, disabled=(status_value == "locked")):
                if not is_active_page:
                    navigate(step_key)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
