import streamlit as st
import pandas as pd

from interface.ui_helpers import _badge, _rec_box

ENCODING_LABELS = {
    "one_hot_encoding":  "One-hot Encoding",
    "label_encoding":    "Label Encoding",
    "ordinal_encoding":  "Ordinal Encoding",
    "drop_column":       "Drop (ตัดออก)",
}

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
| **Ordinal** | Categories มี order จริง (เช่น Low/Mid/High) | รักษา order ไว้ | ต้องมั่นใจว่า order ถูกต้อง |
| **Drop** | ID, Free-text | ลด noise | สูญเสีย feature |
""")

    def on_encoding_change():
        st.session_state["trans_confirmed"] = False
        st.session_state.pop("trans_summary", None)
        st.session_state.pop("transformed_df", None)

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
    {_badge(info["warning"], "orange") if info["warning"] else ""}
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
  <span style="background:#d2992233;color:#d29922;padding:2px 8px;border-radius:4px;font-size:0.78rem;font-weight:700;vertical-align:middle">NOTE</span> <b>กรุณายืนยัน order ที่ถูกต้อง</b><br>
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
            on_change=on_encoding_change,
            horizontal=True,
            label_visibility="collapsed",
        )
        decisions[col] = chosen
        st.markdown("---")

    return decisions