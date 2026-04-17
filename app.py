import streamlit as st
from data_prepare.features.loading_data import load_from_local, load_target_col, delete_local
from data_prepare.features.target_col import suggest_target

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
    """เปลี่ยนหน้าโดยใช้ session_state เป็น source of truth แล้วค่อย sync URL"""
    st.session_state["_step"] = step
    st.query_params["step"] = step
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

    # Recovery Cache
    if st.session_state.get("main_df") is None:
        recovered_df, recover_name = load_from_local()
        if recovered_df is not None:
            st.session_state["main_df"] = recovered_df
            st.session_state["last_uploaded_file"] = recover_name

    # Restore target_col ถ้าหายไปหลัง refresh (session_state ว่าง)
    if "target_col" not in st.session_state and st.session_state.get("main_df") is not None:
        saved = load_target_col()
        main_cols = list(st.session_state["main_df"].columns)
        if saved and saved in main_cols:
            st.session_state["target_col"] = saved
        else:
            suggested, _ = suggest_target(st.session_state["main_df"])
            st.session_state["target_col"] = suggested

    # Navigation — session_state เป็น source of truth, query_params แค่ sync URL
    # ถ้ามี _step ใน session_state ให้ใช้ค่านั้นก่อน (ป้องกัน race condition กับ rerun)
    url_step = st.session_state.get("_step") or st.query_params.get("step", "upload")
    # guard: ถ้าไม่มีข้อมูลแต่พยายามเข้าหน้า inner → กลับ upload
    if url_step in ("cleaning", "eda", "transformation", "model_process") and st.session_state.get("main_df") is None:
        url_step = "upload"
    # sync URL ให้ตรงกับ step จริง
    st.session_state["_step"] = url_step
    if st.query_params.get("step") != url_step:
        st.query_params["step"] = url_step

    is_upload_page = url_step not in ("cleaning", "eda", "transformation", "model_process")

    # แสดงปุ่ม New Dataset เฉพาะหน้าที่ไม่ใช่ Upload
    if is_upload_page:
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
    if url_step == "cleaning":
        from interface.cleaning import render_cleaning
        render_cleaning()
    elif url_step == "eda":
        from interface.eda import render_eda
        render_eda()
    elif url_step == "transformation":
        from interface.data_transformation import render_transformation
        render_transformation()
    elif url_step == "model_process":
        from interface.model_process import render_ml_process
        render_ml_process()
    else:
        from interface.upload import render_upload
        render_upload()


if __name__ == "__main__":
    main()