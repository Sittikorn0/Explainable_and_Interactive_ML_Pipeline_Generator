import streamlit as st
import pandas as pd
import plotly.express as px

SCALING_LABELS = {
    "log_transform":  "Log Transform (log1p)",
    "standard_scaler": "Standard Scaler (Z-score)",
    "minmax_scaler":   "MinMax Scaler [0, 1]",
    "robust_scaler":   "Robust Scaler (Median/IQR)",
    "no_scaling":      "ไม่ทำ Scaling",
}
SCALING_WHEN = {
    "log_transform":   "เมื่อข้อมูล skewed รุนแรง (|skew| > 2) เช่น รายได้ ราคา จำนวน transaction",
    "standard_scaler": "เมื่อข้อมูลกระจายแบบ normal, ไม่มี outlier",
    "minmax_scaler":   "เมื่อข้อมูล skewed เล็กน้อย หรือต้องการช่วง [0,1]",
    "robust_scaler":   "เมื่อมี outlier มาก (>5% ของข้อมูล)",
    "no_scaling":      "เมื่อใช้ tree-based model ล้วนๆ",
}

def render_scaling(dataframe: pd.DataFrame, target_column: str,
                    scaling_analysis: dict) -> str:
    st.markdown('<div class="section-header">SCALING</div>', unsafe_allow_html=True)

    column_statistics = scaling_analysis["column_stats"]
    if not column_statistics:
        st.success("ไม่มี numeric columns ที่ต้องทำ scaling")
        return "no_scaling"

    # แสดง stats ของ numeric columns
    with st.expander("Feature Statistics"):
        statistics_df = pd.DataFrame(column_statistics)
        st.dataframe(
            statistics_df,
            hide_index=True,
            width="stretch",
            column_config={
                "col":         st.column_config.TextColumn("COLUMN"),
                "min":         st.column_config.NumberColumn("MIN",  format="%.3f"),
                "max":         st.column_config.NumberColumn("MAX",  format="%.3f"),
                "mean":        st.column_config.NumberColumn("MEAN", format="%.3f"),
                "std":         st.column_config.NumberColumn("STD",  format="%.3f"),
                "skew":        st.column_config.NumberColumn("SKEW", format="%.3f"),
                "outlier_pct": st.column_config.NumberColumn("OUTLIER %", format="%.1f%%"),
            }
        )

    # Visualization
    numeric_columns = [stats["col"] for stats in column_statistics]
    if numeric_columns:
        st.markdown('<div style="font-family:monospace; font-size:0.7rem; color:#475569; margin: 10px 0 5px 0; letter-spacing: 0.1em;">DISTRIBUTION PLOT</div>', unsafe_allow_html=True)
        selected_column = st.selectbox("Select column to visualize:", numeric_columns, key="sc_viz_col", label_visibility="collapsed")
        figure = px.histogram(dataframe, x=selected_column, nbins=30, marginal="box",
                           color_discrete_sequence=["#7AA2F7"])
        figure.update_layout(template="plotly_dark", height=280, showlegend=False,
                          margin=dict(t=10, b=10, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(figure, width="stretch")

    # Recommendation
    rec_label = SCALING_LABELS.get(scaling_analysis["recommended"], scaling_analysis["recommended"]).upper()
    st.markdown(f"""
<div class="premium-card premium-card-blue" style="padding: 20px 24px !important;">
    <div class="technical-label" style="color: #7AA2F7; margin-bottom: 6px; font-size: 0.8rem; letter-spacing: 0.12em; font-weight: 700;">RECOMMENDED METHOD: {rec_label}</div>
    <div style="font-size: 1rem; color: #94A3B8; line-height: 1.6;">{scaling_analysis["reason"]}</div>
</div>
""", unsafe_allow_html=True)

    # แสดง info พิเศษสำหรับ log transform
    if scaling_analysis["recommended"] == "log_transform":
        heavy_skew_columns = scaling_analysis.get("heavy_skew_cols", [])
        st.markdown(f"""
<div style="border-left: 3px solid #7AA2F7; background: rgba(122, 162, 247, 0.03); padding: 16px 20px; margin: 12px 0; font-family: monospace; font-size: 0.9rem;">
    <div style="color: #7AA2F7; margin-bottom: 6px; letter-spacing: 0.05em; font-weight: 700;">LOG TRANSFORM SPEC:</div>
    <div style="color: #94A3B8; line-height: 1.6;">
        Method: log1p(x)<br>
        Target: {', '.join(heavy_skew_columns[:3])}{'...' if len(heavy_skew_columns)>3 else ''}
    </div>
</div>
""", unsafe_allow_html=True)

    # เลือก method
    with st.expander("METHOD COMPARISON", expanded=False):
        method_rows = []
        for key, label in SCALING_LABELS.items():
            method_rows.append({"Method": label, "Condition": SCALING_WHEN[key]})
        st.table(pd.DataFrame(method_rows))

    available_options = scaling_analysis["options"]
    recommended_value = scaling_analysis["recommended"]
    
    def on_scaling_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    default_index = 0
    if "scaling_method" in st.session_state and st.session_state["scaling_method"] in available_options:
        default_index = available_options.index(st.session_state["scaling_method"])
    else:
        default_index = available_options.index(recommended_value)

    chosen_scaling = st.radio(
        "Select Scaling Method",
        options=available_options,
        format_func=lambda method: SCALING_LABELS.get(method, method),
        index=default_index,
        key="scaling_method",
        on_change=on_scaling_change,
        horizontal=True,
    )
    return chosen_scaling