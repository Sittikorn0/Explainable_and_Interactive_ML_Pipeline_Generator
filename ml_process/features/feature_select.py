import streamlit as st
import pandas as pd
import plotly.express as px

from interface.ui_helpers import _badge, _rec_box


def _render_leakage_check(df: pd.DataFrame, target_col: str) -> list:
    """แสดง leakage warning และให้ user เลือก drop column ที่น่าสงสัย"""
    from ml_process.features.logic import analyze_leakage

    st.subheader("3. Data Leakage Check — ตรวจ column ที่อาจทำให้ model โกง")

    with st.spinner("กำลังตรวจสอบ Data Leakage..."):
        items = analyze_leakage(df, target_col)

    if not items:
        st.success("ไม่พบ column ที่น่าสงสัย — dataset ดูสะอาด")
        return []

    high  = [x for x in items if x["severity"] == "high"]
    color = "#f85149" if high else "#d29922"
    desc  = (f"พบ {len(high)} column ที่มีความเสี่ยงสูง — แนะนำ drop ก่อน Apply Transformation"
             if high else
             f"พบ {len(items)} column ที่น่าสงสัย — ตรวจสอบก่อน Apply Transformation")

    st.markdown(
        f'<div style="background:#1a0f0f;border:1px solid {color};border-radius:10px;'
        f'padding:14px 18px;margin:8px 0">'
        f'<div style="color:{color};font-weight:700;font-size:1rem;margin-bottom:4px">'
        f'พบ column ที่น่าสงสัย ({len(items)} รายการ)</div>'
        f'<div style="color:#c9d1d9;font-size:0.9rem">{desc}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    to_drop = []
    for item in items:
        sev_color = "#f85149" if item["severity"] == "high" else "#d29922"
        sev_label = "HIGH RISK" if item["severity"] == "high" else "MEDIUM RISK"
        reasons_str = " · ".join(item["reasons"])
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'padding:8px 0;border-bottom:1px solid #21262d">'
            f'<span style="background:{sev_color}22;color:{sev_color};font-size:0.75rem;'
            f'font-weight:700;padding:2px 8px;border-radius:4px;flex-shrink:0;margin-top:3px">'
            f'{sev_label}</span>'
            f'<div style="flex:1">'
            f'<code style="color:#e6edf3;font-size:0.95rem">{item["col"]}</code>'
            f'<div style="color:#8b949e;font-size:0.875rem;margin-top:4px">{reasons_str}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.checkbox(
            f"Drop `{item['col']}`",
            value=(item["severity"] == "high"),
            key=f"drop_leakage_{item['col']}",
        ):
            to_drop.append(item["col"])

    return to_drop


def _render_feature_selection(df: pd.DataFrame, target_col: str,
                               fs_analysis: dict) -> list:
    st.subheader("4. Feature Selection — ตัด Features ที่ไม่จำเป็น")

    drop_high_corr = fs_analysis["drop_high_corr"]
    drop_low_var   = fs_analysis["drop_low_var"]

    if not drop_high_corr and not drop_low_var:
        st.success("ไม่พบ features ที่ควรตัดออก")
        return []

    to_drop = []

    def on_fs_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    # High Correlation
    if drop_high_corr:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:16px 0 8px">'
            '<span style="background:#f8514933;color:#f85149;padding:2px 8px;border-radius:4px;'
            'font-size:0.78rem;font-weight:700">HIGH CORR</span>'
            '<span style="font-weight:600;font-size:1rem;color:#e6edf3">'
            'คอลัมน์ที่มี Correlation สูง (≥ 0.85)</span></div>',
            unsafe_allow_html=True,
        )
        _rec_box("ตัดคอลัมน์ที่ซ้ำซ้อนออก", fs_analysis["reason_corr"])

        # Heatmap
        num_cols = [c for c in df.columns
                    if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             range_color=[-1, 1], aspect="auto")
            fig.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")

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
                on_change=on_fs_change,
            ):
                to_drop.append(col_b)

    # Low Variance
    if drop_low_var:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:16px 0 8px">'
            '<span style="background:#d2992233;color:#d29922;padding:2px 8px;border-radius:4px;'
            'font-size:0.78rem;font-weight:700">LOW VAR</span>'
            '<span style="font-weight:600;font-size:1rem;color:#e6edf3">'
            'คอลัมน์ที่มี Variance ต่ำมาก</span></div>',
            unsafe_allow_html=True,
        )
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
                on_change=on_fs_change,
            ):
                to_drop.append(item["col"])

    return list(set(to_drop))