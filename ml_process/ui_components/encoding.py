import streamlit as st
import pandas as pd

from interface.ui_helpers import badge, recommendation_box

ENCODING_LABELS = {
    "one_hot_encoding":  "One-hot Encoding",
    "label_encoding":    "Label Encoding",
    "ordinal_encoding":  "Ordinal Encoding",
    "drop_column":       "Drop (ตัดออก)",
}

def render_encoding(dataframe: pd.DataFrame, target_column: str,
                     encoding_analysis: list) -> dict:
    st.markdown('<div class="section-header">ENCODING</div>', unsafe_allow_html=True)

    if not encoding_analysis:
        st.success("ไม่มี categorical columns ที่ต้องทำ encoding")
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

        st.markdown(f"""
<div class="premium-card premium-card-purple" style="padding: 20px 24px !important;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
        <div class="technical-value" style="color: #BB9AF7; font-size: 1.25rem;">{col_name}</div>
        <div style="font-family: monospace; font-size: 0.85rem; color: #64748b; letter-spacing: 0.05em;">{cardinality_value} UNIQUE</div>
    </div>
    <div style="font-family: monospace; font-size: 0.9rem; color: #475569; margin-bottom: 16px; line-height: 1.4;">
        {sample_values_text}{"..." if len(info["sample_values"]) >= 5 else ""}
    </div>
    <div style="border-top: 1px solid rgba(148, 163, 184, 0.08); padding-top: 14px;">
        <div style="font-family: monospace; font-size: 0.8rem; color: #BB9AF7; margin-bottom: 6px; letter-spacing: 0.12em; font-weight: 700;">RECOMMENDED: {ENCODING_LABELS.get(recommended_method, recommended_method).upper()}</div>
        <div style="font-size: 1rem; color: #94A3B8; line-height: 1.6;">{info["reason"]}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        if info["warning"]:
            st.markdown(f"""
<div style="border-left: 2px solid #F59E0B; background: rgba(245, 158, 11, 0.05); padding: 8px 16px; margin-bottom: 12px; font-size: 0.85rem; color: #F59E0B; font-family: monospace;">
    [!] {info["warning"]}
</div>
""", unsafe_allow_html=True)

        if recommended_method == "ordinal_encoding":
            st.markdown(f"""
<div style="background:#2d1f0a;border:1px solid #d29922;border-radius:6px;
padding:10px 14px;margin:6px 0;font-size:0.81rem;color:#d29922">
  <span style="background:#d2992233;color:#d29922;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:700;vertical-align:middle">NOTE</span> <b>กรุณายืนยัน order ที่ถูกต้อง</b><br>
  <span style="color:#c9d1d9">
  ระบบเรียง alphabetical อัตโนมัติ: 
  <b>{" &lt; ".join(sorted(str(value) for value in info["sample_values"]))}</b><br>
  ถ้า order ไม่ถูกต้อง ให้เลือก Label Encoding แทน
  </span>
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