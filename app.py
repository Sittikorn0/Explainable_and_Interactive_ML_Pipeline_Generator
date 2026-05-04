import streamlit as st
from data_prepare.features.loading_data import load_from_local, load_target_col, delete_local
from data_prepare.features.target_col import suggest_target
from explainable.features.pipeline_state import get_step_status, rollback_to, STEP_ORDER, STEP_LABELS

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
    "ml_process": ml_page,
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
        f'<h2 style="font-family:{SANS};font-size:1.45rem;font-weight:700;'
        f'color:#F1F5F9;margin:0 0 5px;">{title}</h2>'
        + (
            f'<p style="font-family:{SANS};font-size:1rem;color:#94A3B8;margin:0;">{subtitle}</p>'
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


@st.dialog("ยืนยันการย้อนกลับ?")
def confirm_rollback(target_step: str):
    st.warning(f"หากคุณย้อนกลับไปที่ขั้นตอน **{STEP_LABELS[target_step]}** ข้อมูลและการตัดสินใจในขั้นตอนหลังจากนี้จะถูกล้างออกทั้งหมด")
    st.markdown("คุณต้องการดำเนินการต่อหรือไม่?")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("ยืนยันการย้อนกลับ", type="primary", width="stretch"):
            rollback_to(target_step)
            navigate(target_step)
    with c2:
        if st.button("ยกเลิก", width="stretch"):
            st.rerun()


_STEP_ICONS = {
    "upload": "📂",
    "cleaning": "🧹",
    "eda": "📊",
    "transformation": "⚙️",
    "ml_process": "🏆",
    "explainable": "💡"
}

def render_step_indicator(current_pg):
    status = get_step_status()
    
    # CSS สำหรับ Segmented Stepper ที่มีความ Premium และเจาะจงเฉพาะปุ่ม
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
    
    /* เจาะจงเฉพาะปุ่มที่มี key ขึ้นต้นด้วย step_btn_ เท่านั้น */
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

    /* ตกแต่งเฉพาะปุ่มปัจจุบัน */
    .current + div + div[class*="st-key-step_btn_"] button div[data-testid="stMarkdownContainer"] p,
    div[class*="st-key-step_btn_"] button:hover div[data-testid="stMarkdownContainer"] p {
        color: #63B3ED !important;
    }
    
    /* เส้นขีดใต้ชื่อขั้นตอนปัจจุบัน - ใช้ Selector ที่เข้าถึงง่ายขึ้น */
    div[class*="st-key-step_btn_"]:has(button[disabled=""]) button::after {
        /* ป้องกันไม่ให้ขึ้นในปุ่ม locked */
    }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(STEP_ORDER))
    
    for i, step_key in enumerate(STEP_ORDER):
        with cols[i]:
            st_val = status[step_key]
            label = STEP_LABELS[step_key]
            icon = _STEP_ICONS.get(step_key, "•")
            is_active_pg = (pages[step_key] == current_pg)
            node_status = st_val if st_val != "current" else "current"
            
            line_html = ""
            if i > 0: line_html += '<div class="line-segment line-left"></div>'
            if i < len(STEP_ORDER) - 1: line_html += '<div class="line-segment line-right"></div>'
            
            st.markdown(f"""
            <div class="node-container {node_status}">
                {line_html}
                <div class="step-node">{icon}</div>
            </div>
            """, unsafe_allow_html=True)
            
            btn_key = f"step_btn_{step_key}"
            txt_color = "#718096"
            if st_val == "done": txt_color = "#68D391"
            elif st_val == "current": txt_color = "#63B3ED"
            
            # CSS เฉพาะปุ่มแบบรายตัว (เพื่อความแน่นอน)
            st.markdown(f"""
            <style>
            div.st-key-{btn_key} button {{
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
                padding: 0 !important;
            }}
            div.st-key-{btn_key} button div[data-testid="stMarkdownContainer"] p {{
                color: {txt_color} !important;
                font-size: 0.78rem !important;
                font-weight: {"700" if st_val == "current" else "500"} !important;
                text-transform: uppercase !important;
                letter-spacing: 0.03em !important;
            }}
            div.st-key-{btn_key} button:hover div[data-testid="stMarkdownContainer"] p {{
                color: #FFF !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            if st.button(label, key=btn_key, disabled=(st_val == "locked")):
                if st_val == "done" and not is_active_pg:
                    confirm_rollback(step_key)
                elif st_val == "current" and not is_active_pg:
                    navigate(step_key)














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
    color: #F8FAFC !important;
}
[data-testid="stMetricLabel"] * {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
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

    pg = st.navigation(nav_pages, position="hidden")

    if "_step" in st.session_state:
        target_page = pages.get(st.session_state["_step"])
        del st.session_state["_step"]
        if target_page and target_page in nav_pages:
            st.switch_page(target_page)

    # Header section
    title_col, btn_col = st.columns([9, 1])

    with title_col:
        st.title("Explainable & Interactive ML Pipeline Generator")
        st.caption("Data Science 1312414 | Education Only")
    
    if pg != upload_page:
        with btn_col:
            st.markdown("<div style='height:2.8rem'></div>", unsafe_allow_html=True)
            file_name = st.session_state.get("last_uploaded_file", "")

            @st.dialog("เริ่มต้นใหม่?")
            def confirm_reset_dialog():
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
                    confirm_reset_dialog()

    # Render Step Indicator
    render_step_indicator(pg)

    st.divider()



    # CSS เฉพาะปุ่ม New Dataset
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