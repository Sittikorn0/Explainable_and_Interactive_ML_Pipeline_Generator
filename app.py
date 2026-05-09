import streamlit as st
from data_prepare.loading_data import load_from_local, load_target_col, delete_local
from data_prepare.logic.target_col import suggest_target
from explainable.state_manager.pipeline_state import get_step_status, STEP_ORDER, STEP_LABELS

# UI Components & Helpers
from interface.ui_helpers import page_header, SANS_FONT
from interface.app_components.ui_navigation import render_step_indicator
from interface.app_components.ui_dialogs import confirm_rollback, confirm_reset_dialog

# Pages
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

def load_css(file_name):
    with open(file_name) as file_obj:
        st.markdown(f"<style>{file_obj.read()}</style>", unsafe_allow_html=True)

load_css("interface/styles/app.css")

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

def main():
    st.set_page_config(
        page_title="AutoML Senior Project",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # UI Global Styling (Metrics, Cards, etc.)
    st.markdown("""
<style>
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

/* Global Styling for Buttons */
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

    # Data & Cache Restoration
    if st.session_state.get("main_df") is None:
        recovered_dataframe, recovered_filename = load_from_local()
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
            from data_prepare.loading_data import load_transformed_df
            recovered_trans_df = load_transformed_df()
            if recovered_trans_df is not None:
                st.session_state["transformed_df"] = recovered_trans_df
            else:
                st.session_state["transformed_df"] = st.session_state.get("main_df")
        else:
            t_sum, t_tar = load_trans_metadata()
            if t_sum:
                st.session_state["trans_summary"] = t_sum
                st.session_state["_trans_target_saved"] = t_tar
                st.session_state["trans_confirmed"] = True
                from data_prepare.loading_data import load_transformed_df
                recovered_trans_df = load_transformed_df()
                if recovered_trans_df is not None:
                    st.session_state["transformed_df"] = recovered_trans_df
                else:
                    st.session_state["transformed_df"] = st.session_state.get("main_df")

    # Restore cleaning status
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

    # Navigation Setup
    navigation_pages = [upload_page, cleaning_page, eda_page, trans_page, ml_page, explain_page]
    current_page_instance = st.navigation(navigation_pages, position="hidden")

    # Session Guard
    if current_page_instance != upload_page and st.session_state.get("main_df") is None:
        st.switch_page(upload_page)

    if "_step" in st.session_state:
        target_navigation_page = pages_mapping.get(st.session_state["_step"])
        del st.session_state["_step"]
        if target_navigation_page and target_navigation_page in navigation_pages:
            st.switch_page(target_navigation_page)

    # --- Header UI ---
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
        btn_col, _ = st.columns([1.6, 8.4])
        with btn_col:
            st.markdown('<div class="custom-reset-marker"></div>', unsafe_allow_html=True)
            if st.button("New Dataset", width="stretch", key="btn_new_dataset"):
                confirm_reset_dialog()

    # Step Indicator
    render_step_indicator(current_page_instance, pages_mapping, get_step_status(), STEP_ORDER, STEP_LABELS)

    st.divider()

    # Shared Button Styles
    st.markdown("""
        <style>
        div[data-testid="stElementContainer"]:has(.custom-reset-marker) ~ div[data-testid="stElementContainer"] .stButton button {
            background-color: transparent !important; color: #f87171 !important;
            border: 1px solid rgba(248, 113, 113, 0.4) !important; border-radius: 8px !important;
            font-size: 0.85rem !important; font-weight: 600 !important; padding: 0.45rem 1rem !important;
        }
        div[data-testid="stElementContainer"]:has(.custom-reset-marker) ~ div[data-testid="stElementContainer"] .stButton button:hover {
            background-color: #f87171 !important; color: #ffffff !important; border-color: #f87171 !important;
        }
        .stButton button:not([kind="primary"]) {
            background-color: transparent !important; color: #94A3B8 !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important; transition: all 0.2s ease !important;
        }
        .stButton button:not([kind="primary"]):hover {
            background-color: rgba(148, 163, 184, 0.1) !important; color: #e2e8f0 !important; border-color: #94A3B8 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    current_page_instance.run()

if __name__ == "__main__":
    main()