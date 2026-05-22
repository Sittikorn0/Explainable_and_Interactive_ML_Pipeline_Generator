import streamlit as st
from backend.core.session.pipeline_state import STEP_LABELS, rollback_to

@st.dialog("ยืนยันการย้อนกลับ?")
# dialog ยืนยันการ rollback ไปยัง step ที่ระบุ พร้อมล้าง downstream ใช้ใน diff view
def confirm_rollback(target_step_name: str):
    from main import navigate
    st.warning(f"หากคุณย้อนกลับไปที่ขั้นตอน **{STEP_LABELS[target_step_name]}** ข้อมูลและการตัดสินใจในขั้นตอนหลังจากนี้จะถูกล้างออกทั้งหมด")
    st.markdown("คุณต้องการดำเนินการต่อหรือไม่?")
    column_1, column_2 = st.columns(2)
    with column_1:
        if st.button("ยืนยันการย้อนกลับ", type="primary", width="stretch"):
            rollback_to(target_step_name)
            navigate(target_step_name)
    with column_2:
        if st.button("ยกเลิก", width="stretch"):
            st.rerun()

@st.dialog("เริ่มต้นใหม่?")
# dialog ยืนยัน reset session ทั้งหมดเพื่อ upload ไฟล์ใหม่ ใช้ใน main.py และ sidebar
def confirm_reset_dialog():
    from main import reset_session_state
    current_filename = st.session_state.get("last_uploaded_file", "")
    st.markdown(
        f"ข้อมูล **{current_filename}** และการแก้ไขทั้งหมดจะถูกลับ\n\n"
        "คุณต้องการกลับไปอัปโหลด Dataset ใหม่หรือไม่?"
    )
    dialog_col1, dialog_col2 = st.columns(2)
    with dialog_col1:
        if st.button("ยืนยัน", type="primary", width="stretch"):
            reset_session_state()
    with dialog_col2:
        if st.button("ยกเลิก", width="stretch"):
            st.rerun()
