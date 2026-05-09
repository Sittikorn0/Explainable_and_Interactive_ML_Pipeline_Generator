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


def render_step_indicator(current_page):
    step_status = get_step_status()
    
    # 1. CSS สำหรับ Minimal List (No Boxes)
    css_rules = ""
    for index, step_key in enumerate(STEP_ORDER):
        status_value = step_status[step_key]
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


def main():
    st.set_page_config(
        page_title="AutoML Senior Project",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="expanded", # เปิด Sidebar ไว้เป็นค่าเริ่มต้น
    )

    st.markdown("""
<style>
/* Existing Metric styles... */
[data-testid="stMetricValue"] * { font-size: 2.5rem !important; font-weight: 700 !important; color: #7AA2F7 !important; }

/* Premium Technical Cards */
.premium-card {
    background: rgba(30, 41, 59, 0.4) !important;
    border: 1px solid rgba(148, 163, 184, 0.1) !important;
    border-left: 4px solid #7AA2F7 !important;
    border-radius: 4px !important;
    padding: 24px !important;
    margin: 16px 0 !important;
}

.premium-card-purple { border-left-color: #BB9AF7 !important; background: rgba(88, 28, 135, 0.05) !important; }
.premium-card-blue   { border-left-color: #7AA2F7 !important; background: rgba(30, 58, 138, 0.05) !important; }
.premium-card-amber  { border-left-color: #F59E0B !important; background: rgba(120, 53, 15, 0.05) !important; }


.technical-label {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
    font-size: 0.95rem !important;
    color: #94A3B8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 8px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

.section-header {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #F8FAFC !important;
    text-transform: uppercase !important;
    letter-spacing: 0.2em !important;
    margin: 40px 0 24px 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 12px !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
    padding-bottom: 10px !important;
}

.technical-value {
    font-family: 'JetBrains Mono', 'Roboto Mono', monospace !important;
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    color: #F8FAFC !important;
}

/* Global Styling for Export/Download Buttons... */
div[data-testid="stDownloadButton"] button {
    background-color: rgba(139, 92, 246, 0.05) !important;
    color: #A78BFA !important;
    border: 1px solid rgba(139, 92, 246, 0.4) !important;
    border-radius: 8px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 0.53rem 1.2rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
}

div[data-testid="stDownloadButton"] button:hover {
    background-color: #8B5CF6 !important;
    color: #ffffff !important;
    border-color: #8B5CF6 !important;
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.4) !important;
    transform: translateY(-2px) !important;
}
</style>
""", unsafe_allow_html=True)

    from data_prepare.loading.session_manager import get_session_id
    current_sid = get_session_id()
    
    if st.session_state.get("main_df") is None:
        recovered_dataframe, recovered_filename = load_from_local()
        # st.sidebar.info(f"Cache Status: {'Found' if recovered_dataframe is not None else 'Not Found'}")
        if recovered_dataframe is not None:
            st.session_state["main_df"] = recovered_dataframe
            st.session_state["last_uploaded_file"] = recovered_filename

    if st.session_state.get("ml_result") is None:
        from data_prepare.loading_data import load_ml_cache, load_trans_metadata
        ml_res, ml_met, trans_sum, ml_tar, sc_used, leak_warn = load_ml_cache()
        if ml_res is not None:
            st.session_state["ml_result"] = ml_res
            st.session_state["ml_metrics"] = ml_met
            st.session_state["trans_summary"] = trans_sum
            st.session_state["_trans_target_saved"] = ml_tar
            st.session_state["_ml_scaling_used"] = sc_used
            st.session_state["_ml_leakage_warnings"] = leak_warn
            st.session_state["trans_confirmed"] = True
            st.session_state["transformed_df"] = st.session_state.get("main_df")
        else:
            # Fallback to standalone trans metadata if ML hasn't been run yet
            t_sum, t_tar = load_trans_metadata()
            if t_sum:
                st.session_state["trans_summary"] = t_sum
                st.session_state["_trans_target_saved"] = t_tar
                st.session_state["trans_confirmed"] = True
                st.session_state["transformed_df"] = st.session_state.get("main_df")

    # Restore cleaning_confirmed state based on filename
    cur_file = st.session_state.get("last_uploaded_file", "")
    if "_cleaned" in cur_file:
        st.session_state["cleaning_confirmed"] = True

    if "target_col" not in st.session_state and st.session_state.get("main_df") is not None:
        saved_target_col = load_target_col()
        main_columns = list(st.session_state["main_df"].columns)
        if saved_target_col and saved_target_col in main_columns:
            st.session_state["target_col"] = saved_target_col
        else:
            suggested_target_col, _ = suggest_target(st.session_state["main_df"])
            st.session_state["target_col"] = suggested_target_col

    # Always enable all pages to prevent 404 on refresh
    navigation_pages = [upload_page, cleaning_page, eda_page, trans_page, ml_page, explain_page]
    current_page_instance = st.navigation(navigation_pages, position="hidden")

    # Sync UID back to URL if it's missing (crucial for refresh persistence)
    if "user_uuid" in st.session_state and st.query_params.get("uid") != st.session_state["user_uuid"]:
        st.query_params["uid"] = st.session_state["user_uuid"]

    # --- Session Guard: Redirect to upload if no data is found in current session ---
    if current_page_instance != upload_page and st.session_state.get("main_df") is None:
        st.switch_page(upload_page)

    if "_step" in st.session_state:
        target_navigation_page = pages_mapping.get(st.session_state["_step"])
        del st.session_state["_step"]
        if target_navigation_page and target_navigation_page in navigation_pages:
            st.switch_page(target_navigation_page)

    # 1. Header Section
    header_container = st.container()
    with header_container:
        st.markdown(f"""
            <div style="display: flex; flex-direction: column; gap: 2px; margin-bottom: 12px;">
                <h1 style="margin: 0; padding: 0; font-size: 2.2rem; font-weight: 800; color: #7AA2F7; letter-spacing: -0.5px; line-height: 1.2;">
                    Explainable & Interactive ML Pipeline Generator
                </h1>
                <p style="margin: 0; padding: 0; font-size: 1rem; color: #94A3B8; font-weight: 500; opacity: 0.9;">
                    Data Science 1312414 | Education Only
                </p>
            </div>
        """, unsafe_allow_html=True)
    
        if current_page_instance != upload_page:
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

            # Place button in a narrow column to keep it compact
            btn_col, _ = st.columns([1.6, 8.4])
            with btn_col:
                st.markdown('<div class="custom-reset-marker"></div>', unsafe_allow_html=True)
                if st.button("New Dataset", width="stretch", key="btn_new_dataset"):
                    confirm_reset_dialog()

    render_step_indicator(current_page_instance)

    st.divider()

    st.markdown(
        """
        <style>
        /* Target the container that has our marker, then find the sibling container with the button */
        div[data-testid="stElementContainer"]:has(.custom-reset-marker) ~ div[data-testid="stElementContainer"] .stButton button {
            background-color: transparent !important;
            color: #f87171 !important;
            border: 1px solid rgba(248, 113, 113, 0.4) !important;
            border-radius: 8px !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            padding: 0.45rem 1rem !important;
            transition: background-color 0.2s ease, color 0.2s ease !important;
            white-space: nowrap !important;
        }
        div[data-testid="stElementContainer"]:has(.custom-reset-marker) ~ div[data-testid="stElementContainer"] .stButton button:hover {
            background-color: #f87171 !important;
            color: #ffffff !important;
            border-color: #f87171 !important;
        }

        /* Styling for Secondary/Back Buttons */
        .stButton button:not([kind="primary"]) {
            background-color: transparent !important;
            color: #94A3B8 !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            transition: all 0.2s ease !important;
        }
        .stButton button:not([kind="primary"]):hover {
            background-color: rgba(148, 163, 184, 0.1) !important;
            color: #e2e8f0 !important;
            border-color: #94A3B8 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    current_page_instance.run()


if __name__ == "__main__":
    main()