import streamlit as st
import pandas as pd
import plotly.express as px

from interface.ui_helpers import _rec_box

# Label
ENCODING_LABELS = {
    "one_hot_encoding":  "One-hot Encoding",
    "label_encoding":    "Label Encoding",
    "ordinal_encoding":  "Ordinal Encoding",
    "drop_column":       "Drop (ตัดออก)",
}
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

def _render_scaling(df: pd.DataFrame, target_col: str,
                    sc_analysis: dict) -> str:
    st.subheader("2. Scaling — ปรับ Scale ของ Numeric Features")

    col_stats = sc_analysis["column_stats"]
    if not col_stats:
        st.success("ไม่มี numeric columns ที่ต้องทำ scaling")
        return "no_scaling"

    # แสดง stats ของ numeric columns
    with st.expander("ดู Statistics ของ Numeric Columns"):
        stats_df = pd.DataFrame(col_stats)
        st.dataframe(
            stats_df,
            hide_index=True,
            width="stretch",
            column_config={
                "col":         st.column_config.TextColumn("Column"),
                "min":         st.column_config.NumberColumn("Min",  format="%.3f"),
                "max":         st.column_config.NumberColumn("Max",  format="%.3f"),
                "mean":        st.column_config.NumberColumn("Mean", format="%.3f"),
                "std":         st.column_config.NumberColumn("Std",  format="%.3f"),
                "skew":        st.column_config.NumberColumn("Skew", format="%.3f"),
                "outlier_pct": st.column_config.NumberColumn("Outlier %", format="%.1f%%"),
            }
        )

    # Visualization: distribution ของ column แรก
    num_cols = [s["col"] for s in col_stats]
    if num_cols:
        sel_col = st.selectbox("ดู distribution ของคอลัมน์:", num_cols, key="sc_viz_col")
        fig = px.histogram(df, x=sel_col, nbins=30, marginal="box",
                           color_discrete_sequence=["#1a6fa5"])
        fig.update_layout(template="plotly_dark", height=300, showlegend=False,
                          margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

    # Recommendation
    _rec_box(
        SCALING_LABELS.get(sc_analysis["recommended"], sc_analysis["recommended"]),
        sc_analysis["reason"],
    )

    # แสดง info พิเศษสำหรับ log transform
    if sc_analysis["recommended"] == "log_transform":
        heavy_cols = sc_analysis.get("heavy_skew_cols", [])
        st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:6px;
padding:10px 14px;margin:6px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.7">
  <b style="color:#58a6ff">วิธีการทำงานของ Log Transform:</b><br>
  ใช้ <code>log1p(x) = log(x + 1)</code> กับทุก column ที่มีค่า ≥ 0<br>
  ({', '.join(f'<code>{c}</code>' for c in heavy_cols[:3])}{'...' if len(heavy_cols)>3 else ''})<br>
  จากนั้นตามด้วย Standard Scaler เพื่อ normalize<br><br>
  <span style="background:#d2992233;color:#d29922;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:700;vertical-align:middle">NOTE</span> <b style="color:#d29922">หมายเหตุ:</b> ใช้ได้เฉพาะ column ที่มีค่า ≥ 0 เท่านั้น
  column ที่มีค่าติดลบจะไม่ถูก log transform
</div>
""", unsafe_allow_html=True)

    # เลือก method
    with st.expander("เปรียบเทียบ Scaling Methods", expanded=False):
        rows = []
        for k, label in SCALING_LABELS.items():
            rows.append({"Method": label, "ใช้เมื่อ": SCALING_WHEN[k]})
        st.table(pd.DataFrame(rows))

    chosen = st.radio(
        "เลือก Scaling Method",
        options=sc_analysis["options"],
        format_func=lambda x: SCALING_LABELS.get(x, x),
        index=sc_analysis["options"].index(sc_analysis["recommended"]),
        key="scaling_method",
        label_visibility="collapsed",
    )
    return chosen