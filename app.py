import streamlit as st
from data_prepare.features.loading_data import load_from_local, load_target_col, delete_local
from data_prepare.features.target_col import suggest_target

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

pages = {
    "upload": upload_page,
    "cleaning": cleaning_page,
    "eda": eda_page,
    "transformation": trans_page,
    "model_process": ml_page,
    "explainable": explain_page
}

SANS = "'DM Sans','Sarabun',sans-serif"

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css("interface/styles/app.css")


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div style="margin-bottom:1.75rem;">'
        f'<h2 style="font-family:{SANS};font-size:1.35rem;font-weight:600;'
        f'color:#f1f5f9;margin:0 0 5px;">{title}</h2>'
        + (
            f'<p style="font-family:{SANS};font-size:0.9rem;color:#64748b;margin:0;">{subtitle}</p>'
            if subtitle
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def navigate(step: str):
    """เปลี่ยนหน้าโดยใช้ st.switch_page ร่วมกับ session_state"""
    st.session_state["_step"] = step
    st.rerun()


def _reset_session():
    """ล้าง session state + cache ทั้งหมด กลับไปหน้า upload"""
    delete_local()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    navigate("upload")


def main():
    st.set_page_config(
        page_title="AutoML Senior Project",
        page_icon="./icon.png",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Override Streamlit internal metric font (ต้อง inject หลัง set_page_config)
    st.markdown("""
<style>
[data-testid="stMetricValue"] * {
    font-size: 2.25rem !important;
    font-weight: 700 !important;
    line-height: 1.2 !important;
}
[data-testid="stMetricLabel"] * {
    font-size: 1rem !important;
    opacity: 0.8 !important;
}
</style>
""", unsafe_allow_html=True)

    # Recovery Cache
    if st.session_state.get("main_df") is None:
        recovered_df, recover_name = load_from_local()
        if recovered_df is not None:
            st.session_state["main_df"] = recovered_df
            st.session_state["last_uploaded_file"] = recover_name

    # Restore ML results หลัง refresh
    if st.session_state.get("ml_result") is None:
        from data_prepare.features.loading_data import load_ml_cache
        ml_result, ml_metrics, trans_summary, ml_target = load_ml_cache()
        if ml_result is not None:
            st.session_state["ml_result"]          = ml_result
            st.session_state["ml_metrics"]         = ml_metrics
            st.session_state["trans_summary"]      = trans_summary
            st.session_state["ml_task_type"]       = ml_result.get("task_type")
            if ml_target:
                st.session_state["_trans_target_saved"] = ml_target

    # Restore target_col ถ้าหายไปหลัง refresh (session_state ว่าง)
    if "target_col" not in st.session_state and st.session_state.get("main_df") is not None:
        saved = load_target_col()
        main_cols = list(st.session_state["main_df"].columns)
        if saved and saved in main_cols:
            st.session_state["target_col"] = saved
        else:
            suggested, _ = suggest_target(st.session_state["main_df"])
            st.session_state["target_col"] = suggested

    # Navigation
    if st.session_state.get("main_df") is None:
        nav_pages = [upload_page]
    else:
        nav_pages = [upload_page, cleaning_page, eda_page, trans_page, ml_page, explain_page]

    pg = st.navigation(nav_pages)

    if "_step" in st.session_state:
        target_page = pages.get(st.session_state["_step"])
        del st.session_state["_step"]
        if target_page and target_page in nav_pages:
            st.switch_page(target_page)

    # แสดงปุ่ม New Dataset เฉพาะหน้าที่ไม่ใช่ Upload
    if pg == upload_page:
        st.title("Explainable & Interactive ML Pipeline Generator")
        st.caption("Data Science 1312414 | Education Only")
    else:
        title_col, btn_col = st.columns([9, 1])
        with title_col:
            st.title("Explainable & Interactive ML Pipeline Generator")
            st.caption("Data Science 1312414 | Education Only")
        with btn_col:
            st.markdown("<div style='height:2.8rem'></div>", unsafe_allow_html=True)
            file_name = st.session_state.get("last_uploaded_file", "")

            @st.dialog("เริ่มต้นใหม่?")
            def confirm_reset():
                st.markdown(
                    f"ข้อมูล **{file_name}** และการแก้ไขทั้งหมดจะถูกลบ\n\n"
                    "คุณต้องการกลับไปอัปโหลด Dataset ใหม่หรือไม่?"
                )
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("ยืนยัน", type="primary", width="stretch"):
                        _reset_session()
                with c2:
                    if st.button("ยกเลิก", width="stretch"):
                        st.rerun()

            reset_wrap = st.container(key="reset_btn_wrap")
            with reset_wrap:
                if st.button("New Dataset", width="stretch", key="btn_new_dataset"):
                    confirm_reset()

            # CSS เฉพาะปุ่ม New Dataset
            st.markdown(
                """
                <style>
                div[data-testid="stVerticalBlock"]:has(> div[data-testid="element-container"] button[key="btn_new_dataset"]),
                [data-key="reset_btn_wrap"] button,
                div.st-key-reset_btn_wrap button {
                    background: transparent !important;
                    color: #f87171 !important;
                    border: 1px solid rgba(248, 113, 113, 0.35) !important;
                    border-radius: 8px !important;
                    font-size: 0.82rem !important;
                    font-weight: 500 !important;
                    padding: 0.45rem 1rem !important;
                    transition: all 0.2s ease !important;
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

    # ── Render Page ───────────────────────────────────
    pg.run()


if __name__ == "__main__":
    main()