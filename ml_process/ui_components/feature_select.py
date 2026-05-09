import streamlit as st
import pandas as pd
import plotly.express as px

def render_leakage_check(leakage_items: list) -> list:
    """แสดง leakage warning และให้ user เลือก drop column ที่น่าสงสัย"""
    st.markdown('<div class="section-header" style="color: #f85149; border-bottom-color: rgba(248, 81, 73, 0.2);">LEAKAGE CHECK</div>', unsafe_allow_html=True)

    if not leakage_items:
        st.success("ไม่พบ column ที่น่าสงสัย — dataset ดูสะอาด")
        return []

    high_risk_items  = [item for item in leakage_items if item["severity"] == "high"]
    card_class = "premium-card-red" if high_risk_items else "premium-card-amber"
    border_color = "#f85149" if high_risk_items else "#d29922"
    
    title_text = "CRITICAL LEAKAGE DETECTED" if high_risk_items else "POTENTIAL LEAKAGE DETECTED"
    status_text = (f"พบ {len(high_risk_items)} คอลัมน์เสี่ยงสูง — แนะนำตัดออกทันทีเพื่อป้องกันโมเดลจำคำตอบ"
                  if high_risk_items else
                  f"พบ {len(leakage_items)} คอลัมน์น่าสงสัย — โปรดตรวจสอบความเกี่ยวข้องกับ Target")

    st.markdown(f"""
    <div class="premium-card {card_class}" style="padding: 20px 24px !important;">
        <div class="technical-label" style="color: {border_color}; margin-bottom: 6px; font-size: 0.8rem; letter-spacing: 0.12em; font-weight: 700;">{title_text}</div>
        <div style="font-size: 1rem; color: #94A3B8; line-height: 1.6;">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

    columns_to_drop = []
    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    
    for item in leakage_items:
        is_high = item["severity"] == "high"
        severity_color = "#f85149" if is_high else "#d29922"
        bg_color = "rgba(248, 81, 73, 0.1)" if is_high else "rgba(210, 153, 34, 0.1)"
        severity_label = "HIGH RISK" if is_high else "MEDIUM RISK"
        
        reasons_html = "".join([f'<div style="margin-bottom:2px;">• {r}</div>' for r in item["reasons"]])
        
        col_container = st.container()
        with col_container:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.4); border: 1px solid rgba(148, 163, 184, 0.1); 
            border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-family: monospace; font-size: 1.1rem; color: #f8fafc; font-weight: 600;">{item['col']}</div>
                    <div style="background: {bg_color}; color: {severity_color}; font-size: 0.7rem; font-weight: 700; 
                    padding: 3px 10px; border-radius: 4px; border: 1px solid {severity_color}44; letter-spacing: 0.05em;">
                        {severity_label}
                    </div>
                </div>
                <div style="color: #64748b; font-size: 0.85rem; font-family: monospace; line-height: 1.5;">
                    {reasons_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.checkbox(
                f"Drop `{item['col']}` from training set",
                value=is_high,
                key=f"drop_leakage_{item['col']}",
            ):
                columns_to_drop.append(item["col"])
        
        st.markdown('<div style="margin-bottom: 20px;"></div>', unsafe_allow_html=True)

    return columns_to_drop


def render_feature_selection(dataframe: pd.DataFrame, target_column: str,
                                feature_selection_analysis: dict) -> list:
    st.markdown('<div class="section-header">FEATURE SELECTION</div>', unsafe_allow_html=True)

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
        st.markdown(f"""
<div class="premium-card premium-card-amber" style="padding: 20px 24px !important;">
    <div class="technical-label" style="color: #F59E0B; margin-bottom: 6px; font-size: 0.8rem; letter-spacing: 0.12em; font-weight: 700;">HIGH CORRELATION DETECTED</div>
    <div style="font-size: 1rem; color: #94A3B8; line-height: 1.6;">{feature_selection_analysis["reason_corr"]}</div>
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
<div class="premium-card premium-card-amber" style="margin: 24px 0 20px 0 !important; padding: 20px 24px !important;">
    <div class="technical-label" style="color: #F59E0B; margin-bottom: 6px; font-size: 0.8rem; letter-spacing: 0.12em; font-weight: 700;">LOW VAR DETECTED</div>
    <div style="font-size: 1rem; color: #94A3B8; line-height: 1.6;">{feature_selection_analysis["reason_var"]}</div>
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