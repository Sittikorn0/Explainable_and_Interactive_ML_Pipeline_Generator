import streamlit as st
import pandas as pd
import plotly.express as px

from interface.ui_helpers import badge, recommendation_box


def render_leakage_check(dataframe: pd.DataFrame, target_column: str) -> list:
    """แสดง leakage warning และให้ user เลือก drop column ที่น่าสงสัย"""
    from ml_process.logic.logic import analyze_leakage

    st.subheader("3. Data Leakage Check — ตรวจ column ที่อาจทำให้ model โกง")

    with st.spinner("กำลังตรวจสอบ Data Leakage..."):
        leakage_items = analyze_leakage(dataframe, target_column)

    if not leakage_items:
        st.success("ไม่พบ column ที่น่าสงสัย — dataset ดูสะอาด")
        return []

    high_risk_items  = [item for item in leakage_items if item["severity"] == "high"]
    border_color = "#f85149" if high_risk_items else "#d29922"
    description_text  = (f"พบ {len(high_risk_items)} column ที่มีความเสี่ยงสูง — แนะนำ drop ก่อน Apply Transformation"
             if high_risk_items else
             f"พบ {len(leakage_items)} column ที่น่าสงสัย — ตรวจสอบก่อน Apply Transformation")

    st.markdown(
        f'<div style="background:#1a0f0f;border:1px solid {border_color};border-radius:10px;'
        f'padding:14px 18px;margin:8px 0">'
        f'<div style="color:{border_color};font-weight:700;font-size:1rem;margin-bottom:4px">'
        f'พบ column ที่น่าสงสัย ({len(leakage_items)} รายการ)</div>'
        f'<div style="color:#c9d1d9;font-size:0.9rem">{description_text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    columns_to_drop = []
    for item in leakage_items:
        severity_color = "#f85149" if item["severity"] == "high" else "#d29922"
        severity_label = "HIGH RISK" if item["severity"] == "high" else "MEDIUM RISK"
        reasons_text = " · ".join(item["reasons"])
        st.markdown(
            f'<div style="display:flex;gap:10px;align-items:flex-start;'
            f'padding:8px 0;border-bottom:1px solid #21262d">'
            f'<span style="background:{severity_color}22;color:{severity_color};font-size:0.75rem;'
            f'font-weight:700;padding:2px 8px;border-radius:4px;flex-shrink:0;margin-top:3px">'
            f'{severity_label}</span>'
            f'<div style="flex:1">'
            f'<code style="color:#e6edf3;font-size:0.95rem">{item["col"]}</code>'
            f'<div style="color:#8b949e;font-size:0.875rem;margin-top:4px">{reasons_text}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if st.checkbox(
            f"Drop `{item['col']}`",
            value=(item["severity"] == "high"),
            key=f"drop_leakage_{item['col']}",
        ):
            columns_to_drop.append(item["col"])

    return columns_to_drop


def render_feature_selection(dataframe: pd.DataFrame, target_column: str,
                               feature_selection_analysis: dict) -> list:
    st.subheader("4. Feature Selection — ตัด Features ที่ไม่จำเป็น")

    drop_high_correlation = feature_selection_analysis["drop_high_corr"]
    drop_low_variance   = feature_selection_analysis["drop_low_var"]

    if not drop_high_correlation and not drop_low_variance:
        st.success("ไม่พบ features ที่ควรตัดออก")
        return []

    columns_to_drop = []

    def on_feature_selection_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    # High Correlation
    if drop_high_correlation:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:16px 0 8px">'
            '<span style="background:#f8514933;color:#f85149;padding:2px 8px;border-radius:4px;'
            'font-size:0.78rem;font-weight:700">HIGH CORR</span>'
            '<span style="font-weight:600;font-size:1rem;color:#e6edf3">'
            'คอลัมน์ที่มี Correlation สูง (≥ 0.85)</span></div>',
            unsafe_allow_html=True,
        )
        recommendation_box("ตัดคอลัมน์ที่ซ้ำซ้อนออก", feature_selection_analysis["reason_corr"])

        # Heatmap
        numeric_columns = [column for column in dataframe.columns
                    if column != target_column and pd.api.types.is_numeric_dtype(dataframe[column])]
        if len(numeric_columns) >= 2:
            correlation_matrix = dataframe[numeric_columns].corr()
            figure  = px.imshow(correlation_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                             range_color=[-1, 1], aspect="auto")
            figure.update_layout(template="plotly_dark", height=350, margin=dict(t=20, b=20))
            st.plotly_chart(figure, width="stretch")

        for pair_index, pair in enumerate(drop_high_correlation):
            column_a, column_b, correlation_value = pair["col_a"], pair["col_b"], pair["corr"]
            st.markdown(
                f'`{column_a}` ↔ `{column_b}` — correlation = **{correlation_value}** '
                f'{badge("แนะนำตัด " + column_b, "orange")}',
                unsafe_allow_html=True
            )
            if st.checkbox(
                f"ตัด `{column_b}` ออก",
                value=True,
                key=f"drop_corr_{pair_index}_{column_a}_{column_b}",
                on_change=on_feature_selection_change,
            ):
                columns_to_drop.append(column_b)

    # Low Variance
    if drop_low_variance:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;margin:16px 0 8px">'
            '<span style="background:#d2992233;color:#d29922;padding:2px 8px;border-radius:4px;'
            'font-size:0.78rem;font-weight:700">LOW VAR</span>'
            '<span style="font-weight:600;font-size:1rem;color:#e6edf3">'
            'คอลัมน์ที่มี Variance ต่ำมาก</span></div>',
            unsafe_allow_html=True,
        )
        recommendation_box("ตัดคอลัมน์ที่แทบไม่มีข้อมูล", feature_selection_analysis["reason_var"])

        for item in drop_low_variance:
            st.markdown(
                f'`{item["col"]}` — std = **{item["std"]}**, CV = **{item["cv"]}** '
                f'{badge("แนะนำตัด", "orange")}',
                unsafe_allow_html=True
            )
            if st.checkbox(
                f"ตัด `{item['col']}` ออก",
                value=True,
                key=f"drop_var_{item['col']}",
                on_change=on_feature_selection_change,
            ):
                columns_to_drop.append(item["col"])

    return list(set(columns_to_drop))