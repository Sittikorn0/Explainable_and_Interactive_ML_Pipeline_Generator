import streamlit as st
import pandas as pd

from main_compo import badge, recommendation_box

ENCODING_LABELS = {
    "one_hot_encoding": "One-hot Encoding",
    "label_encoding": "Label Encoding",
    "ordinal_encoding": "Ordinal Encoding",
    "drop_column": "Drop (ตัดออก)",
}

# render UI encoding section พร้อม radio selector สำหรับแต่ละ categorical column คืน decisions dict ใช้ใน transformation_page
def render_ml_encoding(dataframe: pd.DataFrame, target_column: str,
                     encoding_analysis: list) -> dict:
    st.markdown('<div class="section-header">ENCODING</div>', unsafe_allow_html=True)

    if not encoding_analysis:
        st.markdown("**[ OK ]** ไม่มี categorical columns ที่ต้องทำ encoding")
        return {}

    with st.expander("Method Comparison", expanded=False):
        st.markdown("""
| Method | เหมาะกับ | ข้อดี | ข้อเสีย |
|---|---|---|---|
| **One-hot** | Cardinality ≤ 10 | ไม่สร้าง ordinal relationship | สร้างหลาย column |
| **Label** | Cardinality > 10, Tree model | ประหยัด column | อาจสร้าง ordinal ที่ไม่มีจริง |
| **Ordinal** | Categories มี order จริง (เช่น Low/Mid/High) | รักษา order ไว้ | ต้องมั่นใจว่า order ถูกต้อง |
| **Drop** | ID, Free-text | ลด noise | สูญเสีย feature |
""")

    def on_encoding_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

    decisions = {}
    for info in encoding_analysis:
        col_name = info["col"]
        cardinality_value = info["cardinality"]
        recommended_method = info["recommended"]
        sample_values_text = ", ".join(str(value) for value in info["sample_values"])
        if len(sample_values_text) > 150:
            sample_values_text = sample_values_text[:150] + "..."

        meta_html = ""
        st.markdown(f"""
<div style="margin-bottom: 24px;">
<div style="margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; align-items: baseline; padding-bottom: 4px; margin-bottom: 4px;">
<span style="font-size: 1.15rem; font-weight: 600; color: #E2E8F0; font-family: monospace;">COLUMN: <span style="color: #BB9AF7;">{col_name}</span></span>
<span style="color: #94A3B8; font-size: 0.85rem; font-family: monospace;">{cardinality_value} UNIQUE</span>
</div>
<div style="color: #94A3B8; font-size: 0.95rem;">
<strong>Sample:</strong> {sample_values_text}
</div>
</div>
<div style="background-color: rgba(187, 154, 247, 0.05); border: 1px solid rgba(187, 154, 247, 0.2); border-radius: 8px; padding: 20px;">
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom: 1px solid rgba(187, 154, 247, 0.1); padding-bottom: 8px; margin-bottom: 8px;">
<div style="color: #BB9AF7; font-weight: bold; font-family: monospace; font-size: 1.05rem;">RECOMMENDED METHOD: {ENCODING_LABELS.get(recommended_method, recommended_method).upper()}</div>
<span style="background:rgba(187,154,247,0.15);color:#BB9AF7;font-size:0.72rem;font-weight:700;padding:2px 7px;border-radius:4px;font-family:monospace">{info['rule_id']}</span>
</div>
<div style="color: #E2E8F0; font-size: 1rem; line-height: 1.6;">{info['reason']}</div>
{meta_html}
</div>
</div>
""", unsafe_allow_html=True)

        if info["warning"]:
            st.markdown(f"""
<div style="background-color: rgba(245, 158, 11, 0.05); border-left: 3px solid #F59E0B; padding: 12px 16px; margin-bottom: 12px; font-size: 0.95rem; color: #E2E8F0;">
<strong style="color: #F59E0B; font-family: monospace;">[!] WARNING</strong><br>
{info['warning']}
</div>
""", unsafe_allow_html=True)

        if recommended_method == "ordinal_encoding":
            st.markdown(f"""
<div style="background-color: rgba(245, 158, 11, 0.05); border-left: 3px solid #F59E0B; padding: 12px 16px; margin-bottom: 12px; font-size: 0.95rem; color: #E2E8F0;">
<strong style="color: #F59E0B; font-family: monospace;">[ NOTE ] กรุณายืนยัน order ที่ถูกต้อง</strong><br>
ระบบเรียง alphabetical อัตโนมัติ: <strong style="font-family: monospace;">{' < '.join(sorted(str(value) for value in info['sample_values']))}</strong><br>
ถ้า order ไม่ถูกต้อง ให้เลือก Label Encoding แทน
</div>
""", unsafe_allow_html=True)

        chosen_method = st.radio(
            f"เลือก method สำหรับ `{col_name}`",
            options=info["options"],
            format_func=lambda method: ENCODING_LABELS.get(method, method),
            index=info["options"].index(recommended_method),
            key=f"enc_{col_name}",
            on_change=on_encoding_change,
            horizontal=True,
            label_visibility="collapsed",
        )
        decisions[col_name] = chosen_method
        st.markdown("---")

    return decisions