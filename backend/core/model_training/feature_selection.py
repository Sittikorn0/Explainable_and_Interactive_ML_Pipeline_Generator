import streamlit as st
import pandas as pd
import plotly.express as px

def render_leakage_check(leakage_items: list) -> list:
    """แสดง leakage warning และให้ user เลือก drop column ที่น่าสงสัย (เฉพาะ High Risk)"""
    st.markdown('<div class="section-header" style="color: #f85149; border-bottom-color: rgba(248, 81, 73, 0.2);">LEAKAGE CHECK</div>', unsafe_allow_html=True)

    # กรองเอาเฉพาะ High Risk ตามที่ผู้ใช้ต้องการ
    high_risk_items = [item for item in leakage_items if item["severity"] == "high"]

    if not high_risk_items:
        st.markdown("**[ OK ]** ไม่พบ High Risk Leakage  dataset พร้อมสำหรับการเทรน")
        return []

    title_text = "CRITICAL LEAKAGE DETECTED"
    status_text = f"พบ {len(high_risk_items)} คอลัมน์เสี่ยงสูง  แนะนำตัดออกทันทีเพื่อป้องกันโมเดลจำคำตอบ"

    st.markdown(f"""
<div style="background-color: rgba(248, 81, 73, 0.05); border: 1px solid rgba(248, 81, 73, 0.3); border-radius: 8px; padding: 20px;">
<div style="color: #f85149; font-weight: bold; font-family: monospace; font-size: 1.05rem; margin-bottom: 8px;">{title_text}</div>
<div style="color: #E2E8F0; font-size: 1rem; line-height: 1.6;">{status_text}</div>
</div>
""", unsafe_allow_html=True)

    columns_to_drop = []
    
    # กรณีมีข้อมูลเยอะ (> 3) -> ใช้ Compact View ใน Expander
    if len(high_risk_items) > 3:
        with st.expander(">> ตรวจสอบรายละเอียดคอลัมน์ที่น่าสงสัยทั้งหมด", expanded=False):
            st.markdown('<div style="margin-bottom: 15px;"></div>', unsafe_allow_html=True)
            
            # 1. แนะนำให้ตัดออก (Multiselect)
            all_leakage_cols = [item["col"] for item in high_risk_items]
            
            selected_cols = st.multiselect(
                "เลือกคอลัมน์ที่จะตัดออก (Drop Columns):",
                options=all_leakage_cols,
                default=all_leakage_cols,
                help="ระบบเลือกคอลัมน์ที่มีความเสี่ยงสูงไว้ให้เป็นค่าเริ่มต้น"
            )
            columns_to_drop.extend(selected_cols)
            
            st.markdown("---")
            st.caption("รายละเอียดเหตุผลประกอบการตัดสินใจ:")
            
            # 2. แสดงตารางเหตุผลแบบกะทัดรัด
            for item in high_risk_items:
                reasons_text = ", ".join([r.replace("**", "") for r in item["reasons"]])
                st.markdown(f"**[!] {item['col']}**: {reasons_text}")
                
    else:
        # กรณีมีข้อมูลน้อย -> ใช้ Detailed View (Card ใหญ่)
        st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
        for item in high_risk_items:
            with st.container():
                reasons_html = "".join([f"<li style='margin-bottom: 4px;'>{r}</li>" for r in item["reasons"]])
                st.markdown(f"""
<div style="background-color: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 8px; margin-bottom: 10px;">
<span style="font-size: 1.1rem; font-weight: 600; color: #f8fafc; font-family: monospace;">{item['col']}</span>
<span style="color: #f85149; font-size: 0.75rem; font-weight: bold; background: rgba(248, 81, 73, 0.1); padding: 2px 8px; border-radius: 4px;">HIGH RISK</span>
</div>
<ul style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5; margin: 0; padding-left: 20px;">
{reasons_html}
</ul>
</div>
""", unsafe_allow_html=True)
                
                if st.checkbox(
                    f"Drop `{item['col']}`",
                    value=True,
                    key=f"drop_leakage_{item['col']}",
                ):
                    columns_to_drop.append(item["col"])
            st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    return columns_to_drop

    return columns_to_drop


def render_ml_feature_selection(dataframe: pd.DataFrame, target_column: str,
                                feature_selection_analysis: dict) -> list:
    st.markdown('<div class="section-header">FEATURE SELECTION</div>', unsafe_allow_html=True)

    drop_high_correlation = feature_selection_analysis["drop_high_corr"]
    drop_low_variance   = feature_selection_analysis["drop_low_var"]

    if not drop_high_correlation and not drop_low_variance:
        st.markdown("**[ OK ]** ไม่พบ features ที่ควรตัดออก")
        return []

    columns_to_drop = []

    def on_feature_selection_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    # High Correlation
    if drop_high_correlation:
        st.markdown(f"""
<div style="background-color: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 20px; margin-bottom: 16px;">
<div style="color: #F59E0B; font-weight: bold; font-family: monospace; font-size: 1.05rem; margin-bottom: 8px;">HIGH CORRELATION DETECTED</div>
<div style="color: #E2E8F0; font-size: 1rem; line-height: 1.6;">{feature_selection_analysis['reason_corr']}</div>
</div>
""", unsafe_allow_html=True)

        # Heatmap
        numeric_columns = [column for column in dataframe.columns
                    if column != target_column and pd.api.types.is_numeric_dtype(dataframe[column])]
        if len(numeric_columns) >= 2:
            correlation_matrix = dataframe[numeric_columns].corr()
            figure  = px.imshow(correlation_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                             range_color=[-1, 1], aspect="auto")
            figure.update_layout(template="plotly_dark", height=320, margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, width="stretch")

        for pair_index, pair in enumerate(drop_high_correlation):
            column_a, column_b, correlation_value = pair["col_a"], pair["col_b"], pair["corr"]
            st.markdown(
                f'<div style="font-family:monospace; font-size:0.95rem; color:#94A3B8; margin-bottom:6px;">'
                f'<code>{column_a}</code> <--> <code>{column_b}</code> | r = **{correlation_value}**'
                f'</div>', unsafe_allow_html=True
            )
            if st.checkbox(
                f"Purge `{column_b}`",
                value=True,
                key=f"drop_corr_{pair_index}_{column_a}_{column_b}",
                on_change=on_feature_selection_change,
            ):
                columns_to_drop.append(column_b)

    # Low Variance
    if drop_low_variance:
        st.markdown(f"""
<div style="background-color: rgba(245, 158, 11, 0.05); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 20px; margin-bottom: 16px;">
<div style="color: #F59E0B; font-weight: bold; font-family: monospace; font-size: 1.05rem; margin-bottom: 8px;">LOW VAR DETECTED</div>
<div style="color: #E2E8F0; font-size: 1rem; line-height: 1.6;">{feature_selection_analysis['reason_var']}</div>
</div>
""", unsafe_allow_html=True)

        for item in drop_low_variance:
            st.markdown(
                f'<div style="font-family:monospace; font-size:0.95rem; color:#94A3B8; margin-bottom:6px;">'
                f'<code>{item["col"]}</code> | std = {item["std"]} | CV = {item["cv"]}'
                f'</div>', unsafe_allow_html=True
            )
            if st.checkbox(
                f"Purge `{item['col']}`",
                value=True,
                key=f"drop_var_{item['col']}",
                on_change=on_feature_selection_change,
            ):
                columns_to_drop.append(item["col"])

    return list(set(columns_to_drop))