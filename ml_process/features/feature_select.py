import streamlit as st
import pandas as pd
import plotly.express as px

from interface.ui_helpers import _badge, _rec_box

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