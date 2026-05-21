# Libraries
import streamlit as st

# Logic Import
from backend.core.session.state import *
from backend.core.session.session_manager import get_session_id
from user_interface.pages.Upload.upload_components.upload_compo import suggest_target
from backend.core.insight.trace_log import *
from backend.core.session.pipeline_state import STEP_ORDER, STEP_LABELS, get_step_status

# UI Import
from user_interface.navigator.step_indicator import render_step_indicator
from user_interface.navigator.dialog import confirm_reset_dialog

# Lazy Loading
def load_upload(): from user_interface.pages.Upload.upload_page import render_upload; render_upload()
def load_cleaning(): from user_interface.pages.Cleaning.cleaning_page import render_cleaning; render_cleaning()
def load_eda(): from user_interface.pages.Eda.eda_page import render_eda; render_eda()
def load_transformation(): from user_interface.pages.Transformation.transformation_page import render_transformation; render_transformation()
def load_model_process(): from user_interface.pages.Model_process.model_process_page import render_model_process; render_model_process()
def load_insight(): from user_interface.pages.Insight.insight_page import render_insight; render_insight()

# Page
upload_page = st.Page(load_upload, title="Upload Dataset", url_path="upload")
cleaning_page = st.Page(load_cleaning, title="Data Cleaning", url_path="cleaning")
eda_page = st.Page(load_eda, title="EDA", url_path="eda")
transformation_page = st.Page(load_transformation, title="Transformation", url_path="transformation")
model_process_page = st.Page(load_model_process, title="Model Process", url_path="model_process")
insight_page = st.Page(load_insight, title="Insight & Explainable", url_path="insight")

all_pages = [upload_page, cleaning_page, eda_page, transformation_page, model_process_page, insight_page]
pages_map = {
    "upload": upload_page,
    "cleaning": cleaning_page,
    "eda": eda_page,
    "transformation": transformation_page,
    "model_process": model_process_page,
    "insight": insight_page
}

# Style
def load_css(path: str) -> None:
    with open(path) as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)

load_css("user_interface/styles/main.css")

def render_main_header() -> None:
    st.markdown("""
        <div style="display:flex;flex-direction:column;gap:2px;margin-bottom:24px;">
            <h1 style="margin:0;padding:0;font-size:2.2rem;font-weight:800;color:#7AA2F7;letter-spacing:-0.5px;line-height:1.2;">
                Explainable and Interactive ML Pipeline Generator
            </h1>
            <div style="font-size:1.05rem;color:#94A3B8;font-weight:500;opacity:0.8;max-width:900px;line-height:1.4;">
                Explainable and Interactive Machine Learning Pipeline Generator for Data Science Education
            </div>
            <div style="font-size:0.85rem;color:#94A3B8;opacity:0.5;margin-top:8px;font-weight:400;">
                Data Science 1312414 | Education Only
            </div>
        </div>
    """, unsafe_allow_html=True)

# Navigate
def navigate(step_name: str):
    """เปลี่ยนหน้าด้วย session_state + rerun"""
    st.session_state["step"] = step_name
    if "user_uuid" in st.session_state:
        st.query_params["uid"] = st.session_state["user_uuid"]
    st.rerun()


def reset_session_state():
    """ล้าง state ทั้งหมดและกลับหน้า upload"""
    delete_local()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    navigate("upload")

# Session
def restore_transformed_df() -> None:
    """โหลด transformed_df จาก disk หรือ fallback เป็น main_df"""
    recovered = load_transformed_df()
    st.session_state["transformed_df"] = recovered if recovered is not None else st.session_state.get("main_df")


def restore_session() -> None:
    """โหลด state ที่บันทึกไว้กลับเข้า session (รันทุกครั้งที่ rerun)"""
    # Restore main dataframe (ข้ามถ้าผู้ใช้เพิ่งล้างข้อมูลด้วย X)
    if st.session_state.get("main_df") is None:
        df, filename = load_from_local()
        if df is not None:
            st.session_state["main_df"] = df
            st.session_state["last_uploaded_file"] = filename

    # Restore ML result + transformation metadata
    if st.session_state.get("ml_result") is None:
        ml_res, ml_met, trans_sum, ml_tar, sc_used, leak_warn = load_ml_cache()
        if ml_res is not None:
            st.session_state["ml_result"] = ml_res
            st.session_state["ml_metrics"] = ml_met
            st.session_state["trans_summary"] = trans_sum
            st.session_state["trans_target_saved"] = ml_tar
            st.session_state["ml_scaling_used"] = sc_used
            st.session_state["ml_leakage_warnings"] = leak_warn
            st.session_state["trans_confirmed"] = True
            restore_transformed_df()
        else:
            t_sum, t_tar = load_trans_metadata()
            if t_sum:
                st.session_state["trans_summary"] = t_sum
                st.session_state["trans_target_saved"] = t_tar
                st.session_state["trans_confirmed"] = True
                restore_transformed_df()

    # Restore cleaning flag
    if "cleaned" in st.session_state.get("last_uploaded_file", ""):
        st.session_state["cleaning_confirmed"] = True

    # Restore target column
    if "target_col" not in st.session_state and st.session_state.get("main_df") is not None:
        cols = list(st.session_state["main_df"].columns)
        # ใช้ non-widget key ก่อน (ไม่ถูก Streamlit ล้างตอน navigate)
        persistent = st.session_state.get("_target_col_persistent")
        if persistent and persistent in cols:
            st.session_state["target_col"] = persistent
        else:
            saved = load_target_col()
            if saved and saved in cols:
                st.session_state["target_col"] = saved
            else:
                suggested, _ = suggest_target(st.session_state["main_df"])
                st.session_state["target_col"] = suggested

# Main
def main():
    st.set_page_config(
        page_title="Explainable ML Pipeline",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation setup must come first so routing is established before
    # get_session_id() potentially triggers st.rerun() to sync the uid.
    current_page = st.navigation(all_pages, position="hidden")

    # Init (get_session_id may rerun; routing is already locked in above)
    get_session_id()
    restore_log()

    # Restore data from disk
    restore_session()

    # Guard: ถ้าไม่มีข้อมูลและไม่ได้อยู่หน้า upload → กลับหน้า upload
    if current_page != upload_page and st.session_state.get("main_df") is None:
        st.switch_page(upload_page)

    # Navigate ถ้ามี step ใน session จาก navigate
    if "step" in st.session_state:
        target = pages_map.get(st.session_state.pop("step"))
        if target and target in all_pages:
            st.switch_page(target)

    # Header
    render_main_header()

    show_new_dataset = (current_page != upload_page) or (st.session_state.get("main_df") is not None)
    if show_new_dataset:
        btn_col, _ = st.columns([1.6, 8.4])
        with btn_col:
            st.markdown('<div class="custom-reset-marker"></div>', unsafe_allow_html=True)
            if st.button("New Dataset", width="stretch", key="btn_new_dataset"):
                confirm_reset_dialog()

    # Step indicator + divider
    render_step_indicator(current_page, pages_map, get_step_status(), STEP_ORDER, STEP_LABELS)
    st.divider()

    # Run current page
    current_page.run()
    
if __name__ == "__main__":
    main()