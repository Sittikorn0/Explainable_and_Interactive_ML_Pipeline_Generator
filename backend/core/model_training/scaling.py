import streamlit as st
import pandas as pd
import plotly.express as px

SCALING_LABELS = {
    "log_transform":   "Log Transform (log1p)",
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


def render_ml_scaling(dataframe: pd.DataFrame, _target_column: str,
                      scaling_analysis: dict) -> dict:
    """
    แสดง UI สำหรับ per-column scaling (ดีไซน์เหมือน Encoding)
    Returns: dict {col: method} ที่ผู้ใช้เลือก
    """
    st.markdown('<div class="section-header">SCALING</div>', unsafe_allow_html=True)

    column_decisions = scaling_analysis.get("column_decisions", [])
    if not column_decisions:
        st.markdown("**[ OK ]** ไม่มี numeric columns ที่ต้องทำ scaling")
        return {}

    options = scaling_analysis.get(
        "options",
        ["log_transform", "standard_scaler", "minmax_scaler", "robust_scaler", "no_scaling"],
    )

    with st.expander("Method Comparison", expanded=False):
        st.markdown("""
| Method | เหมาะกับ | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **Standard Scaler** | ข้อมูล Normal, ไม่มี outlier | รักษา relative distance | ถูก outlier ดึง |
| **MinMax Scaler** | ข้อมูล skewed เล็กน้อย | ช่วง [0,1] ชัดเจน | ไวต่อ outlier |
| **Robust Scaler** | มี outlier มาก | ทนต่อ outlier | ช่วงไม่ fixed |
| **Log Transform** | skewed รุนแรง (>2) | ลด skewness ก่อน scale | ใช้ได้เฉพาะค่า ≥ 0 |
| **ไม่ทำ Scaling** | Tree-based model | ไม่ต้อง preprocess | โมเดลอื่นอาจแย่ลง |
""")

    def on_scaling_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    # Visualization
    st.markdown(
        '<div style="font-family:monospace;font-size:0.7rem;color:#475569;'
        'margin:10px 0 5px 0;letter-spacing:0.1em;">DISTRIBUTION PLOT</div>',
        unsafe_allow_html=True,
    )
    numeric_cols = [d["col"] for d in column_decisions]
    selected_col = st.selectbox(
        "Select column to visualize:", numeric_cols,
        key="sc_viz_col", label_visibility="collapsed",
    )
    fig = px.histogram(dataframe, x=selected_col, nbins=30, marginal="box",
                       color_discrete_sequence=["#7AA2F7"])
    fig.update_layout(
        template="plotly_dark", height=280, showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    scaling_decisions: dict[str, str] = {}

    for decision in column_decisions:
        col        = decision["col"]
        rec        = decision["recommended"]
        rule_id    = decision.get("rule_id", "")
        reason     = decision.get("reason", "")
        skew_val   = decision.get("skew", 0)
        out_pct    = decision.get("outlier_pct", 0)
        min_val    = decision.get("min", 0)
        max_val    = decision.get("max", 0)

        rec_label = SCALING_LABELS.get(rec, rec).upper()

        st.markdown(f"""
<div style="margin-bottom: 24px;">
<div style="margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 4px; margin-bottom: 4px;">
<span style="font-size: 1.15rem; font-weight: 600; color: #E2E8F0; font-family: monospace;">COLUMN: <span style="color: #7AA2F7;">{col}</span></span>
<span style="color: #94A3B8; font-size: 0.85rem; font-family: monospace;">skew: {skew_val:+.2f} &nbsp;|&nbsp; outlier: {out_pct:.1f}% &nbsp;|&nbsp; [{min_val:.2f}, {max_val:.2f}]</span>
</div>
</div>
<div style="background-color: rgba(122, 162, 247, 0.05); border: 1px solid rgba(122, 162, 247, 0.2); border-radius: 8px; padding: 20px;">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid rgba(122, 162, 247, 0.1); padding-bottom: 8px; margin-bottom: 8px;">
<div style="color: #7AA2F7; font-weight: bold; font-family: monospace; font-size: 1.05rem;">RECOMMENDED METHOD: {rec_label}</div>
<span style="background:rgba(122,162,247,0.15);color:#7AA2F7;font-size:0.72rem;font-weight:700;padding:2px 7px;border-radius:4px;font-family:monospace">{rule_id}</span>
</div>
<div style="color: #E2E8F0; font-size: 1rem; line-height: 1.6;">{reason}</div>
</div>
</div>
""", unsafe_allow_html=True)

        widget_key  = f"scl_{col}"
        saved       = st.session_state.get(widget_key)
        default_idx = options.index(saved) if saved in options else (options.index(rec) if rec in options else 0)

        chosen = st.radio(
            f"เลือก method สำหรับ `{col}`",
            options=options,
            format_func=lambda m: SCALING_LABELS.get(m, m),
            index=default_idx,
            key=widget_key,
            on_change=on_scaling_change,
            horizontal=True,
            label_visibility="collapsed",
        )
        scaling_decisions[col] = chosen
        st.markdown("---")

    return scaling_decisions
