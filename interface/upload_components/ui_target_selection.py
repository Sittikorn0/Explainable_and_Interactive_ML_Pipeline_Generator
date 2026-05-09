import streamlit as st
import pandas as pd
from data_prepare.logic.target_col import suggest_target, get_column_reasons, describe_target
from data_prepare.loading_data import save_target_col

def render_target_selection(dataframe: pd.DataFrame):
    """แสดงส่วนการเลือก Target Column และบทวิเคราะห์ที่เกี่ยวข้อง"""
    suggested_column, suggested_reasons = suggest_target(dataframe)

    if "target_col" not in st.session_state:
        st.session_state["target_col"] = suggested_column
    if st.session_state.pop("_revert_target", False):
        st.session_state["target_col"] = suggested_column

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    
    target_col1, target_col2 = st.columns([1.5, 3])
    
    with target_col1:
        st.markdown("""
            <div style="border-left: 3px solid #7AA2F7; padding-left: 15px; margin-bottom: 12px">
                <div style="font-weight:700; font-size:1.25rem; color:#FFFFFF">Select Target</div>
                <div style="font-size:0.85rem; color:#94A3B8">เลือกคอลัมน์ที่ต้องการพยากรณ์</div>
            </div>
        """, unsafe_allow_html=True)
        
        selected_target = st.selectbox(
            "เลือก Target Column",
            options=list(dataframe.columns),
            key="target_col",
            label_visibility="collapsed",
            on_change=lambda: save_target_col(st.session_state["target_col"]),
        )
        
        # --- Metadata Badges ---
        from ml_process.logic.data_analyzer import detect_task
        dtype = str(dataframe[selected_target].dtype)
        unique_count = dataframe[selected_target].nunique()
        task_type = detect_task(dataframe, selected_target)
        
        badge_style = "display:inline-block;padding:2px 10px;border-radius:4px;font-size:0.75rem;font-weight:600;margin-right:6px;margin-top:10px;text-transform:uppercase;letter-spacing:0.02em;"
        st.markdown(f"""
            <div style="margin-top:15px">
                <span style="{badge_style}background:rgba(122, 162, 247, 0.1);color:#7AA2F7;border:1px solid rgba(122, 162, 247, 0.2)">{dtype}</span>
                <span style="{badge_style}background:rgba(187, 154, 247, 0.1);color:#BB9AF7;border:1px solid rgba(187, 154, 247, 0.2)">Unique: {unique_count}</span>
                <span style="{badge_style}background:rgba(158, 206, 106, 0.1);color:#9ECE6A;border:1px solid rgba(158, 206, 106, 0.2)">{task_type}</span>
            </div>
        """, unsafe_allow_html=True)

    with target_col2:
        is_suggested = (selected_target == suggested_column)
        title = "Recommended Analysis" if is_suggested else "Manual Selection Analysis"
        accent_color = "#7AA2F7" if is_suggested else "#E0AF68"
        bg_color = "rgba(122, 162, 247, 0.03)" if is_suggested else "rgba(224, 175, 104, 0.03)"
        border_color = "rgba(122, 162, 247, 0.15)" if is_suggested else "rgba(224, 175, 104, 0.15)"
        
        reasons = suggested_reasons if is_suggested else get_column_reasons(dataframe, selected_target)
        reason_html = "".join(f"<li style='margin-bottom:8px;color:#A9B1D6;font-size:0.95rem'><span style='color:{accent_color};margin-right:8px'>•</span>{r}</li>" for r in reasons)
        
        st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border_color}; border-radius:8px; padding:1.5rem; height:100%">
                <div style="display:flex; align-items:center; gap:12px; margin-bottom:15px">
                    <div style="width:10px; height:10px; border-radius:50%; background:{accent_color}; box-shadow: 0 0 8px {accent_color}66"></div>
                    <span style="font-weight:700; font-size:1rem; color:#FFFFFF; text-transform:uppercase; letter-spacing:0.05em">{title}</span>
                </div>
                <ul style="margin:0; padding:0; list-style:none">
                    {reason_html}
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        if not is_suggested:
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button(f"↩ Revert to Recommended ({suggested_column})", key="revert_target", width="stretch"):
                st.session_state["_revert_target"] = True
                st.rerun()
