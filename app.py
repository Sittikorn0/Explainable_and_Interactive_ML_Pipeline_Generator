import streamlit as st
import os
import glob
from features.loading_data import load_from_local


def _clean_old_cache():
    """ลบไฟล์ temp_cache ที่ไม่ได้เป็นของ session ปัจจุบัน"""
    folder = "temp_cache"
    if not os.path.exists(folder):
        return
    current_uid = st.query_params.get("uid", "")
    for pattern in ["cleaned_*.csv", "temp_*.parquet", "meta_*.txt"]:
        for f in glob.glob(os.path.join(folder, pattern)):
            # ลบถ้าไม่ใช่ไฟล์ของ session ปัจจุบัน
            if current_uid and current_uid in f:
                continue
            try:
                os.remove(f)
            except Exception:
                pass


MONO = "'JetBrains Mono','DM Mono',monospace"
SANS = "'DM Sans','Sarabun',sans-serif"

def load_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
load_css("interface/styles/app.css")

def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div style="margin-bottom:1.75rem;">'
        f'<h2 style="font-family:{SANS};font-size:1.35rem;font-weight:600;'
        f'color:#f1f5f9;margin:0 0 5px;">{title}</h2>'
        + (f'<p style="font-family:{SANS};font-size:0.9rem;color:#64748b;margin:0;">{subtitle}</p>'
           if subtitle else "")
        + '</div>',
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(
        page_title="AutoML Senior Project",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Explainable & Interactive ML Pipeline Generator")
    st.caption("Data Science 1312414 | Education Only", width="stretch")

    _clean_old_cache()

    # Recovery Cache
    if st.session_state.get("main_df") is None:
        recovered_df, recover_name = load_from_local()
        if recovered_df is not None:
            st.session_state["main_df"] = recovered_df
            st.session_state["last_uploaded_file"] = recover_name

    url_step = st.query_params.get("step", "upload")

    # Guard: ถ้าไม่มีข้อมูลให้กลับ upload
    if url_step in ("cleaning", "eda", "model", "transformation", "ml_process")             and st.session_state.get("main_df") is None:
        url_step = "upload"
        st.query_params["step"] = "upload"

    # Render Page
    if url_step == "cleaning":
        from interface.cleaning import render_cleaning
        render_cleaning()
    elif url_step == "eda":
        from interface.eda import render_eda
        render_eda()
    elif url_step == "transformation":
        from ml_process.transformation.page import render_transformation
        render_transformation()
    elif url_step in ("model", "ml_process"):
        from ml_process.code import render_ml_process
        render_ml_process() 
    else:
        from interface.upload import render_upload
        render_upload()


if __name__ == "__main__":
    main()