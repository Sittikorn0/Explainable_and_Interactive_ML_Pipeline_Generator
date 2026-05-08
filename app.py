import streamlit as st
from data_prepare.loading_data import load_from_local, load_target_col, delete_local
from data_prepare.logic.target_col import suggest_target
from explainable.state_manager.pipeline_state import get_step_status, rollback_to, STEP_ORDER, STEP_LABELS

from interface.upload import render_upload
from interface.cleaning import render_cleaning
from interface.eda import render_eda
from interface.data_transformation import render_transformation
from interface.model_process import render_ml_process
from interface.explainable import render_explainable

upload_page = st.Page(render_upload, title="Upload Dataset", url_path="upload")
cleaning_page = st.Page(render_cleaning, title="Data Cleaning", url_path="cleaning")
eda_page = st.Page(render_eda, title="Exploratory Data Analysis", url_path="eda")
trans_page = st.Page(render_transformation, title="Data Transformation", url_path="transformation")
ml_page = st.Page(render_ml_process, title="ML Process & Leaderboard", url_path="model_process")
explain_page = st.Page(render_explainable, title="Explainable & Insights", url_path="explainable")

pages_mapping = {
    "upload": upload_page,
    "cleaning": cleaning_page,
    "eda": eda_page,
    "transformation": trans_page,
    "ml_process": ml_page,
    "explainable": explain_page
}

SANS_FONT = "'DM Sans','Sarabun',sans-serif"

def load_css(file_name):
    with open(file_name) as file_obj:
        st.markdown(f"<style>{file_obj.read()}</style>", unsafe_allow_html=True)

load_css("interface/styles/app.css")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div style="margin-bottom:1.75rem;">'
        f'<h2 style="font-family:{SANS_FONT};font-size:1.45rem;font-weight:700;'
        f'color:#F1F5F9;margin:0 0 5px;">{title}</h2>'
        + (
            f'<p style="font-family:{SANS_FONT};font-size:1rem;color:#94A3B8;margin:0;">{subtitle}</p>'
            if subtitle
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def navigate(step_name: str):
    """เปลี่ยนหน้าโดยใช้ st.switch_page ร่วมกับ session_state"""
    st.session_state["_step"] = step_name
    st.rerun()


def reset_session_state():
    """ล้าง session state และ cache ทั้งหมด กลับไปหน้า upload"""
    delete_local()
    for session_key in list(st.session_state.keys()):
        del st.session_state[session_key]
    navigate("upload")


@st.dialog("ยืนยันการย้อนกลับ?")
def confirm_rollback(target_step_name: str):
    st.warning(f"หากคุณย้อนกลับไปที่ขั้นตอน **{STEP_LABELS[target_step_name]}** ข้อมูลและการตัดสินใจในขั้นตอนหลังจากนี้จะถูกล้างออกทั้งหมด")
    st.markdown("คุณต้องการดำเนินการต่อหรือไม่?")
    column_1, column_2 = st.columns(2)
    with column_1:
        if st.button("ยืนยันการย้อนกลับ", type="primary", width="stretch"):
            rollback_to(target_step_name)
            navigate(target_step_name)
    with column_2:
        if st.button("ยกเลิก", width="stretch"):
            st.rerun()


STEP_ICONS_MAPPING = {
    "upload": "📂",
    "cleaning": "🧹",
    "eda": "📊",
    "transformation": "⚙️",
    "ml_process": "🏆",
    "explainable": "💡"
}

def render_step_indicator(current_page):
    step_status = get_step_status()
    
    st.markdown("""
    <style>
    .stepper-wrapper {
        margin-top: 10px;
        margin-bottom: 25px;
        width: 100%;
    }
    
    [data-testid="stColumn"] > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }

    .node-container {
        position: relative;
        width: 100%;
        height: 44px;
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 0;
    }
    
    .line-segment {
        position: absolute;
        top: 50%;
        height: 2px;
        background: #2D3748;
        z-index: 1;
        transform: translateY(-50%);
        pointer-events: none;
    }
    .line-left { left: -25%; width: 75%; }
    .line-right { right: -25%; width: 75%; }
    
    .step-node {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.1rem;
        border: 2px solid #2D3748;
        background: #0E1117 !important;
        color: #4A5568;
        position: relative;
        z-index: 5;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .done .step-node {
        border-color: #48BB78 !important;
        color: #9AE6B4 !important;
        box-shadow: 0 0 10px rgba(72, 187, 120, 0.1);
    }
    .current .step-node {
        border-color: #4299E1 !important;
        background: #2A4365 !important;
        color: #EBF8FF !important;
        transform: scale(1.15);
        box-shadow: 0 0 20px rgba(66, 153, 225, 0.3);
    }
    .locked .step-node {
        opacity: 0.4;
    }
    
    div[class*="st-key-step_btn_"] button {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        min-height: unset !important;
        height: auto !important;
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        margin-top: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    div[class*="st-key-step_btn_"] button div[data-testid="stMarkdownContainer"] p {
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        margin: 0 !important;
        transition: all 0.3s ease !important;
    }

    .current + div + div[class*="st-key-step_btn_"] button div[data-testid="stMarkdownContainer"] p,
    div[class*="st-key-step_btn_"] button:hover div[data-testid="stMarkdownContainer"] p {
        color: #63B3ED !important;
    }
    
    div[class*="st-key-step_btn_"]:has(button[disabled=""]) button::after {
    }
    </style>
    """, unsafe_allow_html=True)
    
    columns = st.columns(len(STEP_ORDER))
    
    for index, step_key in enumerate(STEP_ORDER):
        with columns[index]:
            status_value = step_status[step_key]
            step_label = STEP_LABELS[step_key]
            step_icon = STEP_ICONS_MAPPING.get(step_key, "•")
            is_active_page = (pages_mapping[step_key] == current_page)
            node_status_class = status_value if status_value != "current" else "current"
            
            line_html_content = ""
            if index > 0: 
                line_html_content += '<div class="line-segment line-left"></div>'
            if index < len(STEP_ORDER) - 1: 
                line_html_content += '<div class="line-segment line-right"></div>'
            
            st.markdown(f"""
            <div class="node-container {node_status_class}">
                {line_html_content}
                <div class="step-node">{step_icon}</div>
            </div>
            """, unsafe_allow_html=True)
            
            button_key = f"step_btn_{step_key}"
            text_color = "#718096"
            if status_value == "done": 
                text_color = "#68D391"
            elif status_value == "current": 
                text_color = "#63B3ED"
            
            st.markdown(f"""
            <style>
            div.st-key-{button_key} button {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
            }}
            div.st-key-{button_key} button div[data-testid="stMarkdownContainer"] p {{
                color: {text_color} !important;
                font-size: 0.78rem !important;
                font-weight: {"700" if status_value == "current" else "500"} !important;
                text-transform: uppercase !important;
                letter-spacing: 0.03em !important;
            }}
            div.st-key-{button_key} button:hover div[data-testid="stMarkdownContainer"] p {{
                color: #FFF !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            if st.button(step_label, key=button_key, disabled=(status_value == "locked")):
                if status_value == "done" and not is_active_page:
                    confirm_rollback(step_key)
                elif status_value == "current" and not is_active_page:
                    navigate(step_key)


def main():
    st.set_page_config(
        page_title="AutoML Senior Project",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
<style>
[data-testid="stMetricValue"] * {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
    color: #F8FAFC !important;
}
[data-testid="stMetricLabel"] * {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
}
</style>
""", unsafe_allow_html=True)

    if st.session_state.get("main_df") is None:
        recovered_dataframe, recovered_filename = load_from_local()
        if recovered_dataframe is not None:
            st.session_state["main_df"] = recovered_dataframe
            st.session_state["last_uploaded_file"] = recovered_filename

    if st.session_state.get("ml_result") is None:
        from data_prepare.loading_data import load_ml_cache
        machine_learning_result, machine_learning_metrics, transformation_summary, machine_learning_target = load_ml_cache()
        if machine_learning_result is not None:
            st.session_state["ml_result"]          = machine_learning_result
            st.session_state["ml_metrics"]         = machine_learning_metrics
            st.session_state["trans_summary"]      = transformation_summary
            st.session_state["ml_task_type"]       = machine_learning_result.get("task_type")
            if machine_learning_target:
                st.session_state["_trans_target_saved"] = machine_learning_target

    if "target_col" not in st.session_state and st.session_state.get("main_df") is not None:
        saved_target_col = load_target_col()
        main_columns = list(st.session_state["main_df"].columns)
        if saved_target_col and saved_target_col in main_columns:
            st.session_state["target_col"] = saved_target_col
        else:
            suggested_target_col, _ = suggest_target(st.session_state["main_df"])
            st.session_state["target_col"] = suggested_target_col

    if st.session_state.get("main_df") is None:
        navigation_pages = [upload_page]
    else:
        navigation_pages = [upload_page, cleaning_page, eda_page, trans_page, ml_page, explain_page]

    current_page_instance = st.navigation(navigation_pages, position="hidden")

    if "_step" in st.session_state:
        target_navigation_page = pages_mapping.get(st.session_state["_step"])
        del st.session_state["_step"]
        if target_navigation_page and target_navigation_page in navigation_pages:
            st.switch_page(target_navigation_page)

    title_column, button_column = st.columns([8.5, 1.5])

    with title_column:
        st.title("Explainable & Interactive ML Pipeline Generator")
        st.caption("Data Science 1312414 | Education Only")
    
    if current_page_instance != upload_page:
        with button_column:
            st.markdown("<div style='height:2.8rem'></div>", unsafe_allow_html=True)
            current_filename = st.session_state.get("last_uploaded_file", "")

            @st.dialog("เริ่มต้นใหม่?")
            def confirm_reset_dialog():
                st.markdown(
                    f"ข้อมูล **{current_filename}** และการแก้ไขทั้งหมดจะถูกลบ\n\n"
                    "คุณต้องการกลับไปอัปโหลด Dataset ใหม่หรือไม่?"
                )
                dialog_col1, dialog_col2 = st.columns(2)
                with dialog_col1:
                    if st.button("ยืนยัน", type="primary", width="stretch"):
                        reset_session_state()
                with dialog_col2:
                    if st.button("ยกเลิก", width="stretch"):
                        st.rerun()

            reset_button_wrap = st.container(key="reset_btn_wrap")
            with reset_button_wrap:
                if st.button("New Dataset", width="stretch", key="btn_new_dataset"):
                    confirm_reset_dialog()

    render_step_indicator(current_page_instance)

    st.divider()

    st.markdown(
        """
        <style>
        div.st-key-reset_btn_wrap button {
            background: transparent !important;
            color: #f87171 !important;
            border: 1px solid rgba(248, 113, 113, 0.35) !important;
            border-radius: 8px !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
            padding: 0.45rem 1rem !important;
            transition: all 0.2s ease !important;
            white-space: nowrap !important;
        }
        div.st-key-reset_btn_wrap button:hover {
            background: rgba(248, 113, 113, 0.12) !important;
            border-color: #f87171 !important;
            color: #fca5a5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    current_page_instance.run()


if __name__ == "__main__":
    main()