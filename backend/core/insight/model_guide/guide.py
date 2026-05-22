# Libraries
import streamlit as st

# UI Import
from user_interface.pages.Insight.insight_components.insight_compo import *

# Functions
MODEL_GUIDE_INFO = {
    "Random Forest": {
        "simple":   "สร้าง Decision Tree หลายต้นพร้อมกัน แล้วให้ทุกต้น vote ผลลัพธ์ เหมือนถามหลายคนแล้วใช้เสียงส่วนใหญ่",
        "strength": "ทนทานต่อข้อมูลที่ผิดปกติ ใช้งานได้ดีกับข้อมูลหลายประเภทโดยไม่ต้อง scale",
        "weakness": "ตีความยากกว่า Decision Tree เดี่ยว ใช้ memory มากขึ้น",
    },
    "Gradient Boosting": {
        "simple":   "สร้าง tree ทีละต้น แต่ละต้นพยายามแก้ข้อผิดพลาดของต้นก่อนหน้า  เรียนรู้จากความผิดพลาดไปเรื่อยๆ",
        "strength": "มักให้ผลแม่นที่สุดกับข้อมูลแบบ tabular",
        "weakness": "ใช้เวลา train นานกว่า ต้องปรับ hyperparameter",
    },
    "Logistic Regression": {
        "simple":   "คำนวณน้ำหนักของแต่ละ feature แล้วรวมเป็นคะแนน บอกได้ทันทีว่า feature ไหนมีผลมากแค่ไหน",
        "strength": "เร็ว ตีความง่าย เหมาะเป็น baseline",
        "weakness": "จับ pattern ซับซ้อนไม่ได้ ต้องการ feature scaling",
    },
    "Decision Tree": {
        "simple":   "ตัดสินใจแบบ if-then-else ทีละขั้น เหมือน flowchart ที่ถามคำถาม Yes/No ไปเรื่อยๆ",
        "strength": "ตีความได้ง่ายที่สุด เห็นเหตุผลการตัดสินใจชัดเจน",
        "weakness": "เสี่ยง overfit ถ้าต้นไม้ลึกเกินไป",
    },
    "XGBoost": {
        "simple":   "Gradient Boosting รุ่นที่ optimize แล้ว มีระบบป้องกัน overfit ในตัว ทำงานได้เร็วกว่า",
        "strength": "accuracy สูงมาก ป้องกัน overfit ได้ดี",
        "weakness": "ต้อง tune hyperparameter หลายตัว",
    },
    "LightGBM": {
        "simple":   "Gradient Boosting ที่เร็วมากเป็นพิเศษ ใช้ algorithm แบบ histogram เพื่อประหยัดเวลาและ memory",
        "strength": "เร็วที่สุดในกลุ่ม boosting เหมาะกับข้อมูลขนาดใหญ่",
        "weakness": "อาจ overfit กับ dataset เล็ก",
    },
    "CatBoost": {
        "simple":   "Gradient Boosting ที่จัดการ categorical features ได้โดยตรง ไม่ต้อง encode เอง",
        "strength": "ใช้ง่าย ลด data leakage จากการ encode",
        "weakness": "ช้ากว่า LightGBM ใช้ memory มาก",
    },
    "Linear Regression": {
        "simple":   "หาสูตรเส้นตรงที่ fit ข้อมูลได้ดีที่สุด  y = w₁×x₁ + w₂×x₂ + ... + c",
        "strength": "ตีความง่ายที่สุด เร็วมาก เห็น coefficient ของแต่ละ feature ทันที",
        "weakness": "จับ pattern โค้งหรือซับซ้อนไม่ได้",
    },
    "SVM": {
        "simple":   "หาเส้นตรง (hyperplane) ที่แบ่ง class ออกจากกัน train ด้วย SGD ทำให้เร็วมากแม้ข้อมูลใหญ่",
        "strength": "เร็วมาก รองรับข้อมูลขนาดใหญ่ได้ดี เหมาะกับ high-dimensional data",
        "weakness": "จับ pattern ที่ไม่เป็นเส้นตรงไม่ได้ ต้องการ feature scaling",
    },
    "kNN": {
        "simple":   "ทำนายโดยดูจาก k ตัวอย่างที่ใกล้เคียงที่สุด  ถ้าเพื่อนบ้าน k คนส่วนใหญ่เป็น class A ก็ทำนายว่าเป็น class A",
        "strength": "ง่ายมาก ไม่ต้อง train",
        "weakness": "ช้ากับข้อมูลใหญ่ ต้องการ scaling",
    },
    "Naive Bayes": {
        "simple":   "คำนวณความน่าจะเป็นว่าข้อมูลนี้อยู่ใน class ไหน โดยสมมติว่าแต่ละ feature กระจายแบบ Normal และไม่ขึ้นต่อกัน",
        "strength": "เร็วมากที่สุด เหมาะเป็น baseline สำหรับข้อมูลตัวเลขที่กระจายใกล้เคียง Normal",
        "weakness": "สมมติว่า feature อิสระกัน ซึ่งมักไม่จริงในทางปฏิบัติ ทำให้แม่นยำน้อยลงกับข้อมูลที่มี feature สัมพันธ์กัน",
    },
}

METRIC_EXPLAIN_INFO = {
    "classification": [
        ("Accuracy",        "#7AA2F7", "% ที่ทำนายถูกจากทั้งหมด  ใช้ได้ดีเมื่อ class ไม่เสียสมดุล", r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}"),
        ("F1(Mac)",         "#9ECE6A", "ค่าเฉลี่ยของ F1 ทุก class  เหมาะกับ class ที่ไม่สมดุล", r"F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}"),
        ("Precision(Mac)",  "#E0AF68", "บรรดาที่ทำนายว่าเป็น class X  มีกี่ % ที่ถูกจริง", r"\text{Precision} = \frac{TP}{TP + FP}"),
        ("Recall(Mac)",     "#BB9AF7", "บรรดาที่เป็น class X จริงๆ  model ตรวจพบได้กี่ %", r"\text{Recall} = \frac{TP}{TP + FN}"),
    ],
    "regression": [
        ("R² Score", "#7AA2F7", "model อธิบาย variance ของข้อมูลได้กี่ % (1.0 = perfect)", r"R^2 = 1 - \frac{\sum_{i} (y_i - \hat{y}_i)^2}{\sum_{i} (y_i - \bar{y})^2}"),
        ("RMSE",     "#9ECE6A", "error เฉลี่ยในหน่วยเดียวกับ target  ยิ่งต่ำยิ่งดี", r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}"),
        ("MSE",      "#E0AF68", "เหมือน RMSE แต่ยกกำลัง 2  error ใหญ่ถูก penalize มากกว่า", r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2"),
    ],
}

CARD_COLUMNS_INFO = [
    ("#7AA2F7", "หลักการ",  "simple"),
    ("#9ECE6A", "จุดแข็ง",  "strength"),
    ("#F7768E", "จุดอ่อน",  "weakness"),
]

# render Model Guide tab (หลักการ/จุดแข็ง/จุดอ่อน ของ best model + metric cards + formula expander) ใช้ใน insight_page
def render_guide(best_model_label: str, task_type: str, metrics_dict: dict):
    model_guide = None
    for key, val in MODEL_GUIDE_INFO.items():
        if key.lower() in best_model_label.lower() or best_model_label.lower() in key.lower():
            model_guide = val
            break

    if model_guide:
        render_section_header(
            f"{best_model_label} ทำงานอย่างไร?",
            "ทำความเข้าใจ model ที่ระบบเลือกให้",
        )
        cards_html_content = "".join(
            f'<div style="flex:1;background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
            f'border-top:3px solid {color};border-radius:{BORDER_RADIUS};padding:{PADDING_STYLE}">'
            f'<div style="color:{color};font-weight:700;font-size:0.9rem;'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">{label}</div>'
            f'<div style="color:{TEXT_COLOR};font-size:1rem;line-height:1.8">{model_guide[field]}</div>'
            f'</div>'
            for color, label, field in CARD_COLUMNS_INFO
        )
        st.markdown(
            f'<div style="display:flex;gap:16px;align-items:stretch">{cards_html_content}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    metric_rows_info = METRIC_EXPLAIN_INFO.get(task_type, [])
    render_section_header(
        "ผลลัพธ์บน Test Set",
        "ข้อมูลที่ model ไม่เคยเห็นตอน train  สะท้อนประสิทธิภาพจริง",
    )

    value_columns = st.columns(len(metric_rows_info))
    for col, (name, color, *_) in zip(value_columns, metric_rows_info):
        metric_value = metrics_dict.get(name, "")
        with col:
            st.markdown(
                f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
                f'border-radius:{BORDER_RADIUS};padding:{PADDING_STYLE};text-align:center">'
                f'<div style="color:{TEXT_DIM_COLOR};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.05em;text-transform:uppercase;margin-bottom:10px">{name}</div>'
                f'<div style="color:{color};font-size:1.8rem;font-weight:800;line-height:1">{metric_value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("คำอธิบายและสูตรคำนวณของแต่ละตัวชี้วัด"):
        for name, color, explanation, formula in metric_rows_info:
            st.markdown(
                f'<div style="border-left: 3px solid {color}; padding-left: 12px; margin-top: 12px;">'
                f'<strong style="color: {color}; font-family: monospace; font-size: 1.05rem;">{name}</strong>'
                f'<p style="color: {TEXT_COLOR}; margin: 4px 0 0 0;">{explanation}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.latex(formula)
