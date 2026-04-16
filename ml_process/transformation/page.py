"""
ml_process/transformation/page.py
Explainable Data Transformation UI
แสดง recommendation + เหตุผล + ให้ user เลือกได้
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from ml_process.transformation.analyzer    import analyze_all
from ml_process.transformation.transformer import apply_all


# ── Labels สำหรับแสดงใน UI ───────────────────────────────────
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


# ── Helper UI ────────────────────────────────────────────────
def _badge(text: str, color: str = "blue") -> str:
    colors = {
        "blue":   ("#1a3a5c", "#58a6ff"),
        "green":  ("#1a3a2a", "#3fb950"),
        "orange": ("#3a2a10", "#d29922"),
        "red":    ("#3a1a1a", "#f85149"),
        "gray":   ("#21262d", "#8b949e"),
    }
    bg, fg = colors.get(color, colors["blue"])
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;border:1px solid {fg}33">{text}</span>'
    )


def _rec_box(title: str, reason: str, warning: str = None):
    """กล่องแสดง recommendation พร้อมเหตุผล"""
    warn_html = ""
    if warning:
        warn_html = (
            f'<div style="margin-top:8px;padding:6px 12px;background:#2d1f0a;'
            f'border-left:3px solid #d29922;border-radius:0 6px 6px 0;'
            f'font-size:0.8rem;color:#d29922">⚠ {warning}</div>'
        )
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-left:4px solid #58a6ff;
border-radius:0 8px 8px 0;padding:12px 16px;margin:8px 0">
  <div style="font-weight:600;color:#58a6ff;font-size:0.88rem;margin-bottom:6px">
     แนะนำ: {title}
  </div>
  <div style="font-size:0.83rem;color:#c9d1d9;line-height:1.7">{reason}</div>
  {warn_html}
</div>
""", unsafe_allow_html=True)


# ── Section 1: Encoding ───────────────────────────────────────
def _render_encoding(df: pd.DataFrame, target_col: str,
                     enc_analysis: list) -> dict:
    st.subheader("1. Encoding — แปลง Categorical เป็นตัวเลข")

    if not enc_analysis:
        st.success("ไม่มี categorical columns ที่ต้องทำ encoding")
        return {}

    # ตารางเปรียบเทียบ method
    with st.expander("เปรียบเทียบ Encoding Methods", expanded=False):
        st.markdown("""
| Method | เหมาะกับ | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **One-hot** | Cardinality ≤ 10 | ไม่สร้าง ordinal relationship | สร้างหลาย column |
| **Label** | Cardinality > 10, Tree model | ประหยัด column | อาจสร้าง ordinal ที่ไม่มีจริง |
| **Drop** | ID, Free-text | ลด noise | สูญเสีย feature |
""")

    decisions = {}
    for info in enc_analysis:
        col         = info["col"]
        cardinality = info["cardinality"]
        recommended = info["recommended"]
        samples     = ", ".join(str(v) for v in info["sample_values"])

        st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:10px 0">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
    <span style="font-family:monospace;font-weight:700;font-size:0.9rem;color:#e6edf3">{col}</span>
    {_badge(f"{cardinality} unique", "gray")}
    {_badge("⚠ " + info["warning"], "orange") if info["warning"] else ""}
  </div>
  <div style="font-size:0.78rem;color:#8b949e;margin-bottom:8px">
    ตัวอย่าง: {samples}{"..." if len(info["sample_values"]) >= 5 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

        _rec_box(
            ENCODING_LABELS.get(recommended, recommended),
            info["reason"],
            info["warning"],
        )

        # แสดง warning พิเศษสำหรับ ordinal encoding
        if recommended == "ordinal_encoding":
            st.markdown(f"""
<div style="background:#2d1f0a;border:1px solid #d29922;border-radius:6px;
padding:10px 14px;margin:6px 0;font-size:0.81rem;color:#d29922">
  ⚠ <b>กรุณายืนยัน order ที่ถูกต้อง</b><br>
  <span style="color:#c9d1d9">
  ระบบเรียง alphabetical อัตโนมัติ: 
  <b>{" &lt; ".join(sorted(str(v) for v in info["sample_values"]))}</b><br>
  ถ้า order ไม่ถูกต้อง ให้เลือก Label Encoding แทน
  </span>
</div>
""", unsafe_allow_html=True)

        chosen = st.radio(
            f"เลือก method สำหรับ `{col}`",
            options=info["options"],
            format_func=lambda x: ENCODING_LABELS.get(x, x),
            index=info["options"].index(recommended),
            key=f"enc_{col}",
            horizontal=True,
            label_visibility="collapsed",
        )
        decisions[col] = chosen
        st.markdown("---")

    return decisions


# ── Section 2: Scaling ────────────────────────────────────────
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
            use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)

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
  <b style="color:#d29922">⚠ หมายเหตุ:</b> ใช้ได้เฉพาะ column ที่มีค่า ≥ 0 เท่านั้น
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


# ── Section 3: Feature Selection ─────────────────────────────
def _render_feature_selection(df: pd.DataFrame, target_col: str,
                               fs_analysis: dict) -> list:
    st.subheader("3. Feature Selection — ตัด Features ที่ไม่จำเป็น")

    drop_high_corr = fs_analysis["drop_high_corr"]
    drop_low_var   = fs_analysis["drop_low_var"]

    if not drop_high_corr and not drop_low_var:
        st.success("ไม่พบ features ที่ควรตัดออก")
        return []

    to_drop = []

    # High Correlation
    if drop_high_corr:
        st.markdown("#### ⚠ คอลัมน์ที่มี Correlation สูง (≥ 0.85)")
        _rec_box("ตัดคอลัมน์ที่ซ้ำซ้อนออก", fs_analysis["reason_corr"])

        # Heatmap
        num_cols = [c for c in df.columns
                    if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             range_color=[-1, 1], aspect="auto")
            fig.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        for pair_idx, pair in enumerate(drop_high_corr):
            col_a, col_b, corr_val = pair["col_a"], pair["col_b"], pair["corr"]
            st.markdown(
                f'`{col_a}` ↔ `{col_b}` — correlation = **{corr_val}** '
                f'{_badge("แนะนำตัด " + col_b, "orange")}',
                unsafe_allow_html=True
            )
            if st.checkbox(
                f"ตัด `{col_b}` ออก",
                value=True,
                key=f"drop_corr_{pair_idx}_{col_a}_{col_b}",
            ):
                to_drop.append(col_b)

    # Low Variance
    if drop_low_var:
        st.markdown("#### ⚠ คอลัมน์ที่มี Variance ต่ำมาก")
        _rec_box("ตัดคอลัมน์ที่แทบไม่มีข้อมูล", fs_analysis["reason_var"])

        for item in drop_low_var:
            st.markdown(
                f'`{item["col"]}` — std = **{item["std"]}**, CV = **{item["cv"]}** '
                f'{_badge("แนะนำตัด", "orange")}',
                unsafe_allow_html=True
            )
            if st.checkbox(
                f"ตัด `{item['col']}` ออก",
                value=True,
                key=f"drop_var_{item['col']}",
            ):
                to_drop.append(item["col"])

    return list(set(to_drop))


# ── Summary ───────────────────────────────────────────────────
def _render_summary(df: pd.DataFrame, transformed_df: pd.DataFrame,
                    summary: dict, target_col: str):
    st.markdown("---")
    st.subheader("สรุปผล Data Transformation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Original Columns",     summary["original_cols"])
    c2.metric("Dropped",              f'-{summary["dropped_cols"]}')
    c3.metric("After Encoding",       summary["final_cols"])
    c4.metric("Scaling",              SCALING_LABELS.get(summary["scaling_method"], "—"))

    with st.expander("ดู Transformed Data (5 rows แรก)"):
        st.dataframe(transformed_df.head(5), use_container_width=True)


# ── Main Render ───────────────────────────────────────────────
def render_transformation():
    from app import page_header

    page_header(
        "Data Transformation",
        "แปลงข้อมูลให้พร้อมสำหรับ ML — ระบบวิเคราะห์และแนะนำวิธีที่เหมาะสมพร้อมเหตุผล",
    )

    # Guard
    if st.session_state.get("main_df") is None:
        st.warning("ไม่พบข้อมูล — กรุณากลับไปทำ Data Preparation ก่อน")
        if st.button("← กลับไป Upload"):
            st.query_params["step"] = "upload"
            st.rerun()
        return

    df        = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(
        f"**Current Dataset:** {file_name}  "
        f"|  {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # ── Target Column (จาก Upload/Cleaning — ไม่แสดง selectbox) ─
    _cols = df.columns.tolist()
    _saved = (st.session_state.get("_trans_target_saved") or
              st.session_state.get("target_col"))
    target_col = _saved if _saved in _cols else _cols[-1]

    st.info(f"**Target Column:** `{target_col}` (เลือกไว้จากขั้นตอน Upload)")
    st.markdown("---")

    # ── วิเคราะห์ข้อมูล (cache ตาม df shape + target) ────────
    cache_key = f"trans_analysis_{df.shape}_{target_col}"
    if st.session_state.get("_trans_cache_key") != cache_key:
        with st.spinner("วิเคราะห์ข้อมูล..."):
            analysis = analyze_all(df, target_col)
        st.session_state["_trans_analysis"]  = analysis
        st.session_state["_trans_cache_key"] = cache_key
    else:
        analysis = st.session_state["_trans_analysis"]

    enc_analysis = analysis["encoding"]
    sc_analysis  = analysis["scaling"]
    fs_analysis  = analysis["feature_selection"]

    # ── Render sections ───────────────────────────────────────
    enc_decisions  = _render_encoding(df, target_col, enc_analysis)
    st.markdown("---")
    scaling_method = _render_scaling(df, target_col, sc_analysis)
    st.markdown("---")
    drop_cols      = _render_feature_selection(df, target_col, fs_analysis)

    # ── Apply + Preview ───────────────────────────────────────
    st.markdown("---")
    if st.button("Apply Transformation",  type="primary", width="stretch"):
        with st.spinner("กำลัง transform..."):
            try:
                transformed_df, summary = apply_all(
                    df, enc_decisions, scaling_method, drop_cols, target_col
                )
                st.session_state["transformed_df"]      = transformed_df
                st.session_state["_trans_target_saved"] = target_col
                st.session_state["trans_summary"]       = summary
                st.session_state["trans_confirmed"]     = True
                st.rerun()
            except Exception as e:
                st.error(f"Transform ล้มเหลว: {e}")
                import traceback
                st.code(traceback.format_exc())

    # แสดง summary ถ้า apply แล้ว
    if st.session_state.get("trans_confirmed"):
        transformed_df = st.session_state["transformed_df"]
        summary        = st.session_state["trans_summary"]
        _render_summary(df, transformed_df, summary, target_col)
        st.success("Transform เสร็จแล้ว — กด Next Step เพื่อไป ML Process")

    # ── Navigation ────────────────────────────────────────────
    st.markdown("---")
    col1, _space, col2 = st.columns([0.8, 8, 0.8])

    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            st.session_state.pop("trans_confirmed", None)
            st.session_state.pop("transformed_df", None)
            st.query_params["step"] = "eda"
            st.rerun()

    with col2:
        confirmed = st.session_state.get("trans_confirmed", False)
        if st.button(
            "Next Step", type="primary", width="stretch",
            disabled=not confirmed,
        ):
            # ส่ง transformed_df ไปให้ ML Process
            st.session_state["main_df"]      = st.session_state["transformed_df"]
            st.session_state["ml_target_col_preset"] = st.session_state.get("_trans_target_saved")
            st.query_params["step"] = "ml_process"
            st.rerun()
        if not confirmed:
            st.caption(
                "กด Apply Transformation ก่อนไปขั้นตอนถัดไป",
                width="content", text_alignment="center",
            )