import streamlit as st
import pandas as pd
from data_prepare.logic.target_col import get_column_reasons, actual_type
from ml_process.logic.data_analyzer import detect_task

def render_target_info(dataframe: pd.DataFrame, target_column: str):
    """แสดงข้อมูล Target Column ในหน้า EDA แบบ Minimalist"""
    
    from data_prepare.logic.target_col import actual_type
    from ml_process.logic.data_analyzer import detect_task
    
    dtype = str(actual_type(dataframe[target_column]))
    unique_count = dataframe[target_column].nunique()
    task_type = detect_task(dataframe, target_column)
    
    # Minimal badge style
    badge = lambda text, color: f'<span style="background:rgba({color}, 0.1); color:rgb({color}); padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-left:8px; border:1px solid rgba({color}, 0.2); text-transform:uppercase;">{text}</span>'
    
    st.markdown(f"""
        <div style="background: rgba(122, 162, 247, 0.05); border: 1px solid rgba(122, 162, 247, 0.12); 
        border-radius: 10px; padding: 20px 28px; margin: 15px 0 30px 0; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 16px; line-height: 1;">
                <span style="color: #7AA2F7; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;">Target:</span>
                <span style="font-family: 'JetBrains Mono', 'Roboto Mono', monospace; font-size: 1.25rem; color: #f8fafc; font-weight: 700;">{target_column}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; line-height: 1;">
                {badge(dtype, "122, 162, 247")}
                {badge(f"Unique: {unique_count}", "187, 154, 247")}
                {badge(task_type, "158, 206, 106")}
            </div>
        </div>
    """, unsafe_allow_html=True)
