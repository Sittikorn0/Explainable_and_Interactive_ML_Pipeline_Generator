# Libraries
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Logic Import
from backend.core.insight.explain.explainer import *
from backend.core.insight.trace_log import get_log
from backend.core.session.pipeline_state import get_comparison, clear_comparison, STEP_LABELS

# UI Import

# Functions
# Design tokens - Midnight Lab Palette (Flat Modern)
BACKGROUND_COLOR       = "#24283B"  # Deep Navy Surface
BORDER_COLOR   = "#414868"          # Sharp Slate Border
TEXT_COLOR     = "#7AA2F7"          # Soft Blue for Headings
TEXT_DIM_COLOR = "#94a3b8"          # Muted Slate
ACCENT_BLUE    = "#7AA2F7"          # Soft Blue
ACCENT_INSIGHT = "#9ECE6A"          # Lime Tea
BORDER_RADIUS        = "4px"        # Minimal Sharp Corner
PADDING_STYLE      = "1.25rem"
MARGIN_GAP      = "margin: 1.5rem 0"

STEP_COLORS_INFO = {
    "Upload":              "#58a6ff",
    "Data Cleaning":       "#3fb950",
    "Data Transformation": "#d29922",
    "Model Process":          "#bc8cff",
}
PIPELINE_STEP_ORDER = ["Upload", "Data Cleaning", "Data Transformation", "Model Process"]

def render_section_header(title: str, subtitle: str = "") -> None:
    subtitle_html = (
        f'<div style="color:{TEXT_DIM_COLOR};font-size:0.95rem;margin-top:4px;font-weight:400;line-height:1.6">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="{MARGIN_GAP}">'
        f'<div style="color:{TEXT_COLOR};font-size:1.3rem;font-weight:700;letter-spacing:-0.01em">{title}</div>'
        f'{subtitle_html}</div>',
        unsafe_allow_html=True,
    )
    
def render_importance(model, X_test, y_test, task_type):
    render_section_header(
        "Feature ไหนสำคัญที่สุด?",
        "ระบบสลับค่าของแต่ละ feature แล้วดูว่า model แย่ลงแค่ไหน  แท่งยาว = สำคัญมาก",
    )

    cache_key = f"_xai_perm_{id(model)}"
    if st.session_state.get(cache_key) is None:
        with st.spinner("กำลังวิเคราะห์ความสำคัญของ feature..."):
            permutation_dataframe = compute_permutation_importance(model, X_test, y_test, task_type)
            st.session_state[cache_key] = permutation_dataframe
    permutation_dataframe = st.session_state[cache_key]

    top_features_count   = min(15, len(permutation_dataframe))
    plot_dataframe = permutation_dataframe.head(top_features_count)

    bar_chart_figure = go.Figure(go.Bar(
        x=plot_dataframe["Importance"],
        y=plot_dataframe["Feature"],
        orientation="h",
        marker=dict(color=plot_dataframe["Importance"].tolist(), colorscale=[[0, "rgba(122, 162, 247, 0.1)"], [1, "#7AA2F7"]], showscale=False),
        text=plot_dataframe["Importance"].round(3).tolist(),
        textposition="outside",
    ))
    bar_chart_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, top_features_count * 36),
        yaxis=dict(autorange="reversed"),
        xaxis_title="ความสำคัญ (ยิ่งสูง = ยิ่งสำคัญ)",
        margin=dict(t=10, b=30, l=10, r=90),
    )
    st.plotly_chart(bar_chart_figure, width="stretch")

    positive_importance_features = permutation_dataframe[permutation_dataframe["Importance"] > 0]
    if len(positive_importance_features) == 0:
        st.warning("ทุก feature มีความสำคัญน้อยมาก  ลองตรวจสอบ dataset อีกครั้ง")
        return

    render_section_header("3 Feature สำคัญที่สุด")

    rank_colors  = ["#E0AF68", "#BB9AF7", "#7AA2F7"]
    rank_labels  = ["#1", "#2", "#3"]
    total_importance = positive_importance_features["Importance"].sum() + 1e-9
    top_3_features = positive_importance_features.head(3).reset_index(drop=True)
    card_columns = st.columns(len(top_3_features))

    for index, col in enumerate(card_columns):
        row_data = top_3_features.iloc[index]
        percentage = row_data["Importance"] / total_importance * 100
        with col:
            st.markdown(
                f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
                f'border-top:3px solid {rank_colors[index]};border-radius:{BORDER_RADIUS};'
                f'padding:20px 16px;text-align:center">'
                f'<div style="color:{rank_colors[index]};font-size:0.85rem;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:10px">{rank_labels[index]}</div>'
                f'<div style="font-family:monospace;color:#e6edf3;font-size:1rem;'
                f'word-break:break-all;margin-bottom:12px">{row_data["Feature"]}</div>'
                f'<div style="color:{rank_colors[index]};font-size:1.6rem;font-weight:800;'
                f'line-height:1">{percentage:.0f}%</div>'
                f'<div style="color:{TEXT_DIM_COLOR};font-size:0.85rem;margin-top:6px">'
                f'ของความสำคัญรวม</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    useless_features = permutation_dataframe[permutation_dataframe["Importance"] <= 0]
    if len(useless_features):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            f"**Feature ที่แทบไม่มีผล ({len(useless_features)} ตัว):**\n"
            + ", ".join(f"`{feature}`" for feature in useless_features["Feature"].head(5))
            + (" ..." if len(useless_features) > 5 else "")
        )

def render_trace():
    render_section_header(
        "บันทึกการตัดสินใจตลอด Pipeline",
        "เจาะลึกทุกร่องรอยการประมวลผล  ตั้งแต่ข้อมูลดิบจนถึงโมเดลที่แม่นยำที่สุด",
    )

    pipeline_log = get_log()
    if not pipeline_log:
        st.info("ยังไม่มีข้อมูล  ระบบจะเริ่มบันทึกเมื่อคุณดำเนินการผ่านแต่ละขั้นตอน")
        return

    # CSS for Timeline
    st.markdown("""<style>
.trace-container { position: relative; padding-left: 45px; margin-top: 20px; }
.trace-line { position: absolute; left: 20px; top: 0; bottom: 0; width: 2px; background: linear-gradient(180deg, #58a6ff 0%, #30363d 100%); z-index: 0; }
.step-card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 30px; position: relative; box-shadow: 0 4px 20px rgba(0,0,0,0.2); }
.step-dot { position: absolute; left: -35px; top: 20px; width: 24px; height: 24px; background: #0d1117; border: 3px solid #58a6ff; border-radius: 50%; z-index: 1; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #58a6ff; font-weight: 700; }
.stat-badge { background: rgba(139, 148, 158, 0.1); border: 1px solid rgba(139, 148, 158, 0.2); border-radius: 6px; padding: 6px 12px; margin-right: 8px; margin-bottom: 8px; display: inline-block; }
.insight-bubble { background: rgba(88, 166, 255, 0.05); border-left: 3px solid #58a6ff; border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 10px 0; }
</style>""", unsafe_allow_html=True)

    st.markdown('<div class="trace-container"><div class="trace-line"></div>', unsafe_allow_html=True)

    for index, entry in enumerate(pipeline_log):
        step_name = entry.get("step", "")
        log_items = entry.get("items", [])
        explanations = entry.get("explanations", [])
        color = STEP_COLORS_INFO.get(step_name, "#8b949e")

        # แยกส่วน Metrics ออกจากข้อมูลอื่นใน log_items
        metrics_html = '<div style="display: flex; flex-wrap: wrap; margin-bottom: 12px;">'
        other_items_html = ""
        
        for item in log_items:
            if ":" in item and "→" in item: # น่าจะเป็น Metric เช่น "Rows: 100 -> 80"
                label, val = item.split(":", 1)
                metrics_html += f'<div class="stat-badge"><div style="font-size: 0.75rem; color: #8b949e; text-transform: uppercase;">{label.strip()}</div><div style="font-size: 1rem; color: #e6edf3; font-weight: 600;">{val.strip()}</div></div>'
            elif not item.startswith("  "): # Header item
                other_items_html += f'<div style="font-weight: 600; color: #c9d1d9; margin: 8px 0 4px;">{item}</div>'
            else: # Detail item
                other_items_html += f'<div style="font-size: 0.85rem; color: #8b949e; padding-left: 12px; border-left: 1px solid #30363d; margin: 2px 0;">{item.strip()}</div>'
        
        metrics_html += "</div>"

        # ปรับปรุงส่วน Insights (Explanations)
        insights_html = ""
        if explanations:
            import re as _re
            def _highlight_col(text: str) -> str:
                """แปลง [col] → badge สีฟ้า ตัวใหญ่"""
                return _re.sub(
                    r'\[([^\]]+)\]',
                    r'<span style="color:#7AA2F7;font-size:1.05rem;font-weight:700;font-family:monospace;">[\1]</span>',
                    text,
                )
            items_content = ""
            for exp in explanations:
                display_text = exp.replace("ทำไม", "").replace("?", "").strip()
                if exp.startswith("ทำไม"):
                    items_content += f'<div style="font-weight: 700; color: #58a6ff; font-size: 0.95rem; margin-bottom: 4px;">{_highlight_col(display_text)}</div>'
                elif exp.startswith("  →"):
                    items_content += f'<div style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; margin-bottom: 8px;">{_highlight_col(exp.strip().replace("→", ""))}</div>'
                elif exp.startswith("หลักการ:"):
                    body = exp[len("หลักการ:"):].strip()
                    items_content += (
                        f'<div style="margin: 14px 0 4px 0; background: rgba(88,166,255,0.08); '
                        f'border: 1px solid rgba(88,166,255,0.35); border-left: 4px solid #58a6ff; '
                        f'border-radius: 6px; padding: 10px 14px;">'
                        f'<span style="color:#e6edf3; font-size:0.92rem; line-height:1.65;">'
                        f'<b style="color:#7AA2F7;">หลักการ:</b> {_highlight_col(body)}</span></div>'
                    )
                elif "✓" in exp or "⚠" in exp or "ℹ" in exp:
                    items_content += f'<div style="font-size: 0.88rem; color: #8b949e; margin: 4px 0;">{_highlight_col(exp)}</div>'
                else:
                    items_content += f'<div style="font-size: 0.9rem; color: #e6edf3; margin: 4px 0;">{_highlight_col(exp)}</div>'

            insights_html = f'<div style="margin-top: 20px; border-top: 1px solid #30363d; padding-top: 16px;"><div style="font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;">Data Insights & Rationale</div><div class="insight-bubble">{items_content}</div></div>'

        # เรนเดอร์ Card
        st.markdown(f'<div class="step-card"><div class="step-dot" style="border-color: {color}; color: {color}">{index + 1}</div><div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;"><span style="font-size: 1.1rem; font-weight: 700; color: {color}; text-transform: uppercase; letter-spacing: 1px;">{step_name}</span></div>{metrics_html}{other_items_html}{insights_html}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_comparison():
    comparison_data = get_comparison()
    if not comparison_data:
        st.info("ยังไม่มีข้อมูลการเปรียบเทียบ (ข้อมูลจะปรากฏขึ้นหากคุณกดย้อนกลับไปแก้ไขขั้นตอนก่อนหน้าแล้วทำใหม่จนเสร็จ)")
        return

    render_section_header(
        "Pipeline Comparison (Split View)",
        "เปรียบเทียบความแตกต่างแบบฝั่งซ้าย (เดิม) และฝั่งขวา (ใหม่)",
    )

    def flatten_dictionary(dictionary_data, prefix=""):
        items_dict = {}
        for key, value in dictionary_data.items():
            if isinstance(value, dict):
                items_dict.update(flatten_dictionary(value, prefix=f"{key}_"))
            else:
                items_dict[prefix + key] = value
        return items_dict

    for step_key in ["upload", "cleaning", "transformation", "model_process"]:
        if step_key not in comparison_data:
            continue
            
        step_data = comparison_data[step_key]
        previous_config = step_data.get("prev") or {}
        current_config = step_data.get("curr") or {}
        
        flat_previous = flatten_dictionary(previous_config)
        flat_current = flatten_dictionary(current_config)
        all_keys_list = sorted(list(set(flat_previous.keys()) | set(flat_current.keys())))

        # GitHub Header
        st.markdown(f"""
        <div style="background:#161B22; padding:8px 16px; border:1px solid #30363D; border-radius:6px 6px 0 0; margin-top:24px; border-bottom:1px solid #30363D;">
            <span style="color:#8B949E; font-family:monospace; font-size:0.85rem;">
                pipeline / <span style="color:#C9D1D9; font-weight:600;">{step_key}.cfg</span>
                <span style="margin-left:8px; background:#21262D; padding:2px 8px; border-radius:10px; font-size:0.75rem; color:#7D8590;">
                    {STEP_LABELS.get(step_key, step_key)}
                </span>
            </span>
        </div>
        <div style="border:1px solid #30363D; border-top:none; border-radius:0 0 6px 6px; overflow:hidden; background:#0D1117;">
            <div style="display:grid; grid-template-columns: 1fr 1fr; border-bottom:1px solid #30363D;">
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; border-right:1px solid #30363D; background:#161B22; font-weight:600;">ORIGINAL CONFIG</div>
                <div style="padding:4px 12px; color:#8B949E; font-size:0.7rem; background:#161B22; font-weight:600;">CURRENT CONFIG</div>
            </div>
        """, unsafe_allow_html=True)
        
        diff_rows_html = []
        for key in all_keys_list:
            value_previous = flat_previous.get(key, "")
            value_current = flat_current.get(key, "")
            is_changed = str(value_previous) != str(value_current)
            
            left_background  = "background:rgba(248, 81, 73, 0.12);" if is_changed else "background:transparent;"
            right_background = "background:rgba(63, 185, 80, 0.15);" if is_changed else "background:transparent;"
            
            left_text_color  = "#F85149" if is_changed else "#8B949E"
            right_text_color = "#3FB950" if is_changed else "#C9D1D9"
            
            diff_rows_html.append(f"""
            <div style="display:grid; grid-template-columns: 1fr 1fr; font-family:monospace; font-size:0.82rem; border-bottom:1px solid #21262D;">
                <div style="{left_background} color:{left_text_color}; padding:6px 12px; border-right:1px solid #30363D; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"-" if is_changed else " "}</span>{key}: {value_previous}
                </div>
                <div style="{right_background} color:{right_text_color}; padding:6px 12px; word-break:break-all;">
                    <span style="opacity:0.5; margin-right:8px;">{"+" if is_changed else " "}</span>{key}: {value_current}
                </div>
            </div>
            """)
        
        st.markdown("".join(diff_rows_html), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Clear History", width="stretch"):
        clear_comparison()
        st.rerun()


# ── Model Characteristics ──────────────────────────────────────────────────────

MODEL_CHARACTERISTICS = {
    # Classification
    "logistic_regression":         {"tags": ["interpretable", "fast"], "pros": ["อธิบายได้ง่ายด้วย coefficients", "train เร็ว ใช้ memory น้อย", "เหมาะกับ binary classification"], "cons": ["สมมติว่าความสัมพันธ์เป็นเชิงเส้น", "อ่อนไหวต่อ outlier และ correlated features"]},
    "decision_tree":               {"tags": ["interpretable", "fast"], "pros": ["อ่านเหมือน flowchart เข้าใจง่ายที่สุด", "ไม่ต้อง scaling", "จัดการ non-linear ได้"], "cons": ["Overfit ง่ายถ้าไม่ prune", "ผลลัพธ์ไม่ stable เมื่อข้อมูลเปลี่ยนเล็กน้อย"]},
    "random_forest":               {"tags": ["basic"], "pros": ["แม่นยำสูง ทนต่อ outlier", "ไม่ต้อง scaling", "บอก feature importance ได้"], "cons": ["อธิบายได้ยากกว่า single tree", "ใช้ memory และเวลา train มากกว่า"]},
    "gradient_boosting":           {"tags": ["accurate", "stable"], "pros": ["แม่นยำสูงมาก จัดการ pattern ซับซ้อนได้", "ทนต่อ noise ได้ดี"], "cons": ["Train ช้ากว่า Random Forest", "ต้องปรับ hyperparameter หลายตัว"]},
    "svm":                         {"tags": ["accurate"], "pros": ["ทำงานได้ดีใน high-dimensional space", "ทนต่อ outlier ปานกลาง"], "cons": ["Train ช้ามากกับข้อมูลขนาดใหญ่", "ต้องทำ scaling ก่อนเสมอ", "อธิบายผลได้ยาก"]},
    "knn":                         {"tags": ["interpretable", "fast"], "pros": ["ง่าย ไม่มี training phase จริงๆ", "เข้าใจง่าย: ทำนายจากเพื่อนบ้านที่ใกล้ที่สุด"], "cons": ["ช้ามากกับข้อมูลขนาดใหญ่", "อ่อนไหวต่อ scale และ noise"]},
    "naive_bayes":                 {"tags": ["fast", "interpretable"], "pros": ["Train และ predict เร็วมาก", "ทำงานได้ดีกับข้อมูลน้อย"], "cons": ["สมมติว่า features เป็นอิสระต่อกัน (ซึ่งมักไม่จริง)", "แม่นยำต่ำกว่า ensemble methods"]},
    "xgboost":                     {"tags": ["accurate", "stable"], "pros": ["แม่นยำสูงมาก มี regularization ในตัว", "จัดการ missing values ได้เอง"], "cons": ["ต้องปรับ hyperparameter ค่อนข้างเยอะ", "อธิบายผลได้ยาก"]},
    "lightgbm":                    {"tags": ["accurate", "fast"], "pros": ["Train เร็วมาก รองรับข้อมูลขนาดใหญ่", "แม่นยำสูงใกล้เคียง XGBoost"], "cons": ["อาจ overfit กับข้อมูลขนาดเล็ก", "parameter มีผลต่อ performance มาก"]},
    "catboost":                    {"tags": ["easy"], "pros": ["จัดการ categorical features ได้เองโดยไม่ต้อง encoding", "ทนต่อ overfitting"], "cons": ["Train ช้ากว่า LightGBM", "ใช้ memory มาก"]},
    # Regression
    "linear_regression":           {"tags": ["interpretable", "fast"], "pros": ["อธิบายได้ด้วย coefficients ชัดเจน", "Train เร็วมาก", "เหมาะกับความสัมพันธ์เชิงเส้น"], "cons": ["ใช้ได้เฉพาะ linear relationship", "อ่อนไหวต่อ outlier"]},
    "decision_tree_regressor":     {"tags": ["interpretable", "fast"], "pros": ["อ่านผลง่าย ไม่ต้อง scaling", "จัดการ non-linear ได้"], "cons": ["Overfit ง่าย", "ผลไม่ smooth"]},
    "random_forest_regressor":     {"tags": ["basic"], "pros": ["แม่นยำสูง ทนต่อ noise", "ไม่ต้อง scaling"], "cons": ["ใช้ memory มาก", "ช้ากว่า single model"]},
    "gradient_boosting_regressor": {"tags": ["accurate", "stable"], "pros": ["แม่นยำสูงมาก จัดการ pattern ซับซ้อน"], "cons": ["Train ช้า ต้องปรับ hyperparameter"]},
    "knn_regressor":               {"tags": ["interpretable", "fast"], "pros": ["ง่าย เข้าใจง่าย"], "cons": ["ช้ากับข้อมูลใหญ่ อ่อนไหวต่อ scale"]},
    "xgboost_regressor":           {"tags": ["accurate", "stable"], "pros": ["แม่นยำสูง มี regularization"], "cons": ["ต้องปรับ hyperparameter เยอะ"]},
    "lightgbm_regressor":          {"tags": ["accurate", "fast"], "pros": ["Train เร็ว รองรับข้อมูลใหญ่"], "cons": ["อาจ overfit กับข้อมูลเล็ก"]},
    "catboost_regressor":          {"tags": ["easy"], "pros": ["จัดการ categorical ได้เอง ทนต่อ overfitting"], "cons": ["Train ช้า ใช้ memory มาก"]},
}

TAG_META = {
    "interpretable": {"label": "อธิบายได้",  "color": "#9ECE6A", "desc": "มีกลไกที่มนุษย์อ่านและตีความได้ง่าย"},
    "accurate":      {"label": "แม่นยำสูง",   "color": "#7AA2F7", "desc": "CV Score สูงสุดในกลุ่ม"},
    "fast":          {"label": "เร็ว/เบา",     "color": "#E0AF68", "desc": "Train และ predict เร็ว เหมาะกับ production"},
    "stable":        {"label": "เสถียร",       "color": "#BB9AF7", "desc": "±Std ต่ำ ให้ผลสม่ำเสมอ"},
    "easy":          {"label": "ใช้งานง่าย",   "color": "#FF9F1C", "desc": "ใช้งานง่าย ไม่ซับซ้อน ไม่ต้องแปลงข้อมูลมาก"},
    "basic":         {"label": "พื้นฐาน",     "color": "#9CA3AF", "desc": "โมเดลพื้นฐาน เข้าใจง่าย"},
}


@st.dialog(" ")
def show_model_explanation_dialog(model_label: str):
    """แสดง Popup อธิบายโมเดลเลียนแบบแท็บ Model Guide"""
    from backend.core.insight.model_guide.guide import MODEL_GUIDE_INFO
    
    # ค้นหาคำอธิบายที่ตรงกันในคู่มือฐานข้อมูล
    model_guide = None
    for key, val in MODEL_GUIDE_INFO.items():
        if key.lower() in model_label.lower() or model_label.lower() in key.lower():
            model_guide = val
            break
            
    if model_guide:
        # Determine the model category
        lbl_lower = model_label.lower()
        if "lightgbm" in lbl_lower:
            category = "GRADIENT BOOSTING"
        elif "xgboost" in lbl_lower:
            category = "GRADIENT BOOSTING"
        elif "catboost" in lbl_lower:
            category = "GRADIENT BOOSTING"
        elif "gradient boosting" in lbl_lower or "gbt" in lbl_lower:
            category = "GRADIENT BOOSTING"
        elif "random forest" in lbl_lower or "rf" in lbl_lower:
            category = "ENSEMBLE LEARNING"
        elif "decision tree" in lbl_lower or "dt" in lbl_lower:
            category = "DECISION TREE"
        elif "logistic" in lbl_lower or "linear regression" in lbl_lower or "lasso" in lbl_lower or "ridge" in lbl_lower or "linear" in lbl_lower:
            category = "LINEAR MODEL"
        elif "knn" in lbl_lower or "k-nearest" in lbl_lower or "nearest neighbor" in lbl_lower:
            category = "NEIGHBOR MODEL"
        elif "svm" in lbl_lower or "support vector" in lbl_lower or "svc" in lbl_lower or "svr" in lbl_lower:
            category = "SUPPORT VECTOR MACHINE"
        elif "naive bayes" in lbl_lower or "gaussiannb" in lbl_lower:
            category = "PROBABILISTIC MODEL"
        else:
            category = "MACHINE LEARNING"

        # Inject CSS style overrides for premium aesthetic matching the mockup
        st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Sarabun:wght@400;500;600;700&display=swap');

/* Apply clean fonts to everything inside the dialog */
div[role="dialog"] * {
    font-family: 'Outfit', 'Inter', 'Sarabun', sans-serif !important;
}

/* Hide default dialog header title but keep close button */
[data-testid="stDialogHeader"] h2,
div[role="dialog"] h2 {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Style default close button to match premium mockup */
[data-testid="stDialogHeader"] button {
    background-color: #0E1420 !important;
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    color: #9CA3AF !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
    position: absolute !important;
    right: 24px !important;
    top: 24px !important;
    z-index: 1000 !important;
}
[data-testid="stDialogHeader"] button:hover {
    border-color: #3B82F6 !important;
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}

/* Style Streamlit secondary buttons inside dialog (e.g., footer button) */
div[role="dialog"] button[kind="secondary"] {
    background-color: #0E1420 !important;
    color: #94A3B8 !important;
    border: 1px solid #1E293B !important;
    border-radius: 6px !important;
    font-size: 0.9rem !important;
    padding: 8px 24px !important;
    transition: all 0.2s ease !important;
}
div[role="dialog"] button[kind="secondary"]:hover {
    border-color: #3B82F6 !important;
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}

/* Custom cards with left-border accent and subtle background gradient */
.card-principle {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.08) 0%, rgba(17, 24, 39, 0) 100%), #111827 !important;
    border: 1px solid #1E293B !important;
    border-left: 5px solid #3B82F6 !important;
    border-radius: 6px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    font-family: sans-serif !important;
}
.card-strengths {
    background: linear-gradient(90deg, rgba(16, 185, 129, 0.08) 0%, rgba(17, 24, 39, 0) 100%), #111827 !important;
    border: 1px solid #1E293B !important;
    border-left: 5px solid #10B981 !important;
    border-radius: 6px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    font-family: sans-serif !important;
}
.card-weaknesses {
    background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, rgba(17, 24, 39, 0) 100%), #111827 !important;
    border: 1px solid #1E293B !important;
    border-left: 5px solid #EF4444 !important;
    border-radius: 6px !important;
    padding: 18px 20px !important;
    margin-bottom: 24px !important;
    font-family: sans-serif !important;
}

/* Header highlight text colors inside cards */
.card-principle .card-header {
    color: #3B82F6 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
}
.card-strengths .card-header {
    color: #10B981 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
}
.card-weaknesses .card-header {
    color: #EF4444 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 8px !important;
    text-transform: uppercase !important;
}

.card-body {
    color: #E2E8F0 !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}
</style>
""", unsafe_allow_html=True)

        # Dialog Header Section
        st.markdown(f"""
<div style="margin-bottom: 24px; font-family: sans-serif;">
    <div style="color: #64748B; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;">MODEL INSIGHT</div>
    <div style="color: #FFFFFF; font-size: 1.5rem; font-weight: 700; line-height: 1.3; margin-bottom: 20px;">คำอธิบายและแนวคิดการทำงานของโมเดล</div>
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="color: #00E676; font-size: 1.8rem; font-weight: 700; font-family: sans-serif; letter-spacing: -0.02em;">{model_label}</span>
        <span style="border: 1px solid #334155; border-radius: 4px; padding: 4px 10px; font-size: 0.7rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; background-color: rgba(30, 41, 59, 0.4);">{category}</span>
    </div>
</div>
""", unsafe_allow_html=True)

        # Content Cards (Principle, Strengths, Weaknesses)
        st.markdown(f"""
<div class="card-principle">
    <div class="card-header">PRINCIPLE • หลักการทำงาน</div>
    <div class="card-body">{model_guide["simple"]}</div>
</div>
<div class="card-strengths">
    <div class="card-header">STRENGTHS • จุดแข็ง</div>
    <div class="card-body">{model_guide["strength"]}</div>
</div>
<div class="card-weaknesses">
    <div class="card-header">WEAKNESSES • จุดอ่อน</div>
    <div class="card-body">{model_guide["weakness"]}</div>
</div>
<div style="border-bottom: 1px solid #1E293B; margin-bottom: 20px;"></div>
""", unsafe_allow_html=True)
    else:
        st.warning(f"ไม่พบข้อมูลอธิบายทางเทคนิคสำหรับโมเดล '{model_label}'")
        
    col1, col2 = st.columns([3.5, 1.2])
    with col2:
        if st.button("ปิดหน้าต่าง", key="close_guide_dialog", use_container_width=True):
            st.rerun()


def render_leaderboard_insight(competition_result: dict):
    """Leaderboard พร้อม Filter ตาม Characteristic และกล่องแสดง Popup ความรู้"""
    competition = competition_result["competition"]
    best_key    = competition_result["best_key"]
    task_type   = competition_result["task_type"]
    metric_name = "Accuracy" if task_type == "classification" else "R² Score"

    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )
    if not ranked:
        st.info("ยังไม่มีผลการแข่งขัน")
        return

    # ── คำนวณ Holdout Test Metrics เชิงลึกแบบ On-the-fly หากยังไม่ได้คำนวณและแคชไว้ ──
    missing_eval_keys = [k for k, v in ranked if k != best_key and st.session_state.get(f"_eval_cache_{k}") is None]
    if missing_eval_keys:
        dataframe = st.session_state.get("transformed_df", st.session_state.get("main_df"))
        preset_target = (st.session_state.get("_trans_target_saved") or st.session_state.get("target_col"))
        if dataframe is not None and preset_target is not None:
            from backend.core.model_training.trainer.train_model import get_model_map
            from backend.core.model_training.evaluation.eval import get_metrics
            from backend.core.model_training.preprocess.pipeline import preprocess
            from sklearn.utils.class_weight import compute_sample_weight
            from sklearn.preprocessing import LabelEncoder
            
            with st.spinner("กำลังประมวลผลและคำนวณ Holdout Test score ของโมเดลผู้ท้าชิง..."):
                try:
                    transformation_summary = st.session_state.get("trans_summary", {})
                    scaling_decisions = transformation_summary.get("scaling_decisions") or None
                    scaling_method    = transformation_summary.get("scaling_method", "standard_scaler")
                    encoding_decisions = transformation_summary.get("encoding_decisions") or None
                    X_tr, X_te, y_tr, y_te, _ = preprocess(
                        dataframe, preset_target,
                        scaling_method=scaling_method,
                        scaling_decisions=scaling_decisions,
                        missing_rules=st.session_state.get("missing_rules"),
                        outlier_rules=st.session_state.get("outlier_rules"),
                        encoding_decisions=encoding_decisions,
                    )
                    
                    model_map = get_model_map()
                    for m_key in missing_eval_keys:
                        try:
                            if m_key in model_map:
                                sel_model = model_map[m_key]()
                                sel_params = competition[m_key].get("best_params", {})
                                if sel_params:
                                    try:
                                        sel_model.set_params(**sel_params)
                                    except Exception:
                                        pass
                                
                                fit_kw = {}
                                if task_type == "classification" and m_key in ("gradient_boosting", "xgboost"):
                                    fit_kw["sample_weight"] = compute_sample_weight("balanced", y_tr)
                                sel_model.fit(X_tr, y_tr, **fit_kw)
                                sel_pred = sel_model.predict(X_te)
                                
                                if task_type == "classification":
                                    le = LabelEncoder()
                                    le.fit(y_tr)
                                    if hasattr(le, "classes_"):
                                        sel_pred = le.inverse_transform(sel_pred)
                                        
                                sel_metrics = get_metrics(y_te, sel_pred, task_type)
                                st.session_state[f"_eval_cache_{m_key}"] = {
                                    "y_pred": sel_pred,
                                    "y_test": y_te,
                                    "metrics": sel_metrics
                                }
                        except Exception as e:
                            pass
                except Exception as err:
                    pass

    render_section_header(
        "Model Leaderboard",
        "เปรียบเทียบโมเดลทั้งหมด และคลิกดูแนวคิดจุดเด่นจุดอ่อนของแต่ละโมเดลเชิงลึก",
    )

    # ── กรองข้อมูลล่วงหน้าเพื่อทำตาราง Grid หรือ Stack ──
    filtered_ranked = list(ranked)

    # ── สไตล์ชีต CSS ขั้นสูงสำหรับประกอบการ์ด HTML และปุ่มลิงก์มินิมอล ──
    st.markdown("""
    <style>
    /* ดึงตัวกล่อง Element Container ทั้งหมดของปุ่มขึ้นไปซ้อนทับในการ์ดหลัก (ทำงานระดับ Block-level ป้องกัน CSS flexbox block) */
    div[data-testid="column"]:has(.minimal-btn-trigger) .element-container:has(div.stButton),
    div[data-testid="column"]:has(.minimal-btn-trigger) div.stButton,
    div[data-testid="column"]:has(.minimal-btn-trigger) div:has(> div.stButton) {
        margin-top: -40px !important; /* ดึงขึ้นมาระดับ 40px เพื่อความกึ่งกลางที่สมมาตร */
        margin-bottom: 16px !important; /* เพิ่มระยะห่างแนวตั้งระหว่างแถวที่สวยสมบูรณ์แบบ */
        display: flex !important;
        justify-content: flex-start !important; /* บังคับตัวครอบแม่ชั้นนอกสุดให้จัดวางแบบชิดซ้าย */
        width: 100% !important;
    }
    .element-container:has(.minimal-btn-trigger) + .element-container {
        margin-top: -40px !important;
        margin-bottom: 16px !important;
        display: flex !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }
    
    /* ปรับปุ่ม st.button ให้กลายเป็นลิงก์ข้อความมินิมอลฝังตัวในการ์ด */
    div[data-testid="column"]:has(.minimal-btn-trigger) div.stButton > button,
    .element-container:has(.minimal-btn-trigger) + .element-container div.stButton > button,
    .minimal-btn-trigger + div div.stButton > button {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: #768390 !important;
        font-size: 0.85rem !important;
        text-align: left !important;
        display: inline-flex !important;
        justify-content: flex-start !important;
        align-items: center !important;
        box-shadow: none !important;
        margin-top: 0 !important; /* เปลี่ยนเป็น 0 เพื่อป้องกันการดึงทับซ้อน (Compounding Margin) ที่ทำให้ปุ่มเกยแถบสี */
        margin-left: 24px !important;
        position: relative !important;
        z-index: 10 !important;
        min-height: unset !important;
        line-height: 1.2 !important;
        font-weight: 500 !important;
        transition: color 0.2s ease, transform 0.2s ease !important;
        width: auto !important; /* บังคับขนาดให้กะทัดรัดพอดีกับข้อความ ไม่ยืดเต็ม 100% ของความกว้าง เพื่อเลี่ยงปัญหากลาง */
    }
    
    /* บังคับเนื้อหาด้านในปุ่มทุกตัวให้ชิดซ้าย 100% เพื่อล้างสไตล์การจัดกึ่งกลางของ Streamlit */
    div[data-testid="column"]:has(.minimal-btn-trigger) div.stButton > button * {
        text-align: left !important;
        justify-content: flex-start !important;
        margin-left: 0 !important;
        margin-right: auto !important;
    }
    div[data-testid="column"]:has(.minimal-btn-trigger) div.stButton > button:hover,
    .element-container:has(.minimal-btn-trigger) + .element-container div.stButton > button:hover,
    .minimal-btn-trigger + div div.stButton > button:hover {
        background: transparent !important;
        text-decoration: underline !important;
        transform: translateX(3px) !important;
    }
    div[data-testid="column"]:has(.rank-btn-01) div.stButton > button:hover,
    .element-container:has(.rank-btn-01) + .element-container div.stButton > button:hover,
    .rank-btn-01 + div div.stButton > button:hover {
        color: #00E676 !important;
    }
    div[data-testid="column"]:has(.rank-btn-02) div.stButton > button:hover,
    .element-container:has(.rank-btn-02) + .element-container div.stButton > button:hover,
    .rank-btn-02 + div div.stButton > button:hover {
        color: #7AA2F7 !important;
    }
    div[data-testid="column"]:has(.rank-btn-03) div.stButton > button:hover,
    .element-container:has(.rank-btn-03) + .element-container div.stButton > button:hover,
    .rank-btn-03 + div div.stButton > button:hover {
        color: #FF9F1C !important;
    }
    div[data-testid="column"]:has(.rank-btn-default) div.stButton > button:hover,
    .element-container:has(.rank-btn-default) + .element-container div.stButton > button:hover,
    .rank-btn-default + div div.stButton > button:hover {
        color: #94A3B8 !important;
    }
    div[data-testid="column"]:has(.minimal-btn-trigger) div.stButton > button:active,
    .element-container:has(.minimal-btn-trigger) + .element-container div.stButton > button:active,
    .minimal-btn-trigger + div div.stButton > button:active {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Leaderboard Table Stack (Vertical Stack matching Mockup, centered using columns) ──
    st.markdown("<br>", unsafe_allow_html=True)
    best_score = ranked[0][1]["cv_score"]
    
    col_left, col_mid, col_right = st.columns([0.4, 9.2, 0.4])
    with col_mid:
        for row_idx in range(0, len(filtered_ranked), 2):
            grid_cols = st.columns(2)
            for col_offset in range(2):
                item_idx = row_idx + col_offset
                if item_idx >= len(filtered_ranked):
                    break
                
                key, val = filtered_ranked[item_idx]
                is_best   = key == best_key
                score_pct = val["cv_score"] / (best_score + 1e-9) * 100
                bar_width = max(10, int(score_pct))

                chars = MODEL_CHARACTERISTICS.get(key, {})
                model_tags = chars.get("tags", [])

                # กำหนดตัวประเมินประสิทธิภาพใน Test Set
                eval_key = "Accuracy" if task_type == "classification" else "R² Score"
                evaluation_metrics = st.session_state.get("ml_metrics", {})
                
                cache_key = f"_eval_cache_{key}"
                test_score_val = None
                if st.session_state.get(cache_key) is not None:
                    test_score_val = st.session_state[cache_key]["metrics"].get(eval_key, None)
                elif key == best_key:
                    test_score_val = evaluation_metrics.get(eval_key, None)

                if test_score_val is not None:
                    test_score_str = f"{test_score_val:.4f}"
                else:
                    test_score_str = "—"

                # ค้นหาอันดับเดิมในรายการ ranked เพื่อระบุชุดโทเค็นสีและการจัดวาง
                original_rank_i = next(i for i, (k, _) in enumerate(ranked) if k == key)
                
                # ชุดธีมสีตามลำดับลีดเดอร์บอร์ด (สีสันพรีเมียมตาม Mockup)
                if original_rank_i == 0:
                    rank_color = "#00E676"  # Neon green
                    bar_color = "#00E676"
                    border_style = "border: 1.5px solid #00E676; box-shadow: 0 0 20px rgba(0, 230, 118, 0.15);"
                    card_bg = "#0A101C"     # Deep premium dark navy
                    btn_class = "rank-btn-01"
                    bar_style = f"background: #00E676; width: {score_pct}%; height: 8px; border-radius: 6px; box-shadow: 0 0 8px #00E676;"
                elif original_rank_i == 1:
                    rank_color = "#4F6B94"  # Elegant slate blue
                    bar_color = "#4F6B94"
                    border_style = "border: 1px solid #1E293B;"
                    card_bg = "#0A101C"
                    btn_class = "rank-btn-02"
                    bar_style = f"background: #4F6B94; width: {score_pct}%; height: 8px; border-radius: 6px;"
                elif original_rank_i == 2:
                    rank_color = "#FF9F1C"  # Vibrant warm orange
                    bar_color = "#FF9F1C"
                    border_style = "border: 1px solid #1E293B;"
                    card_bg = "#0A101C"
                    btn_class = "rank-btn-03"
                    bar_style = f"background: #FF9F1C; width: {score_pct}%; height: 8px; border-radius: 6px;"
                else:
                    rank_color = "#475569"  # Muted grey
                    bar_color = "#334155"
                    border_style = "border: 1px solid #161B22;"
                    card_bg = "#060B14"
                    btn_class = "rank-btn-default"
                    bar_style = f"background: #334155; width: {score_pct}%; height: 8px; border-radius: 6px;"

                # กำหนดสีป้าย Badge ตามสีของ Rank เพื่อความสวยงามคุมโทนเหมือน Mockup
                tag_badges = "".join(
                    f'<span style="background:transparent; color:{rank_color}; '
                    f'border:1px solid {rank_color}66; font-size:0.8rem; font-weight:500; '
                    f'padding:4px 12px; border-radius:8px; margin-right:8px; white-space:nowrap;">{TAG_META[t]["label"]}</span>'
                    for t in model_tags if t in TAG_META
                )
                
                best_badge = (
                    f'<span style="background:#00E676; color:#06150D; font-weight:800; '
                    f'font-size:0.7rem; padding:2px 8px; border-radius:4px; margin-left:8px; '
                    f'letter-spacing:0.05em; vertical-align:middle; line-height:1;">BEST</span>'
                    if is_best else ""
                )

                with grid_cols[col_offset]:
                    # แสดงกล่องการ์ด HTML โครงสร้างกล่องเดี่ยวไร้รอยต่อ (เขียนแบบชิดซ้ายเพื่อป้องกัน Markdown Parse Error)
                    st.markdown(f"""<div style="background: {card_bg}; {border_style} border-radius: 16px; padding: 24px 24px 24px 24px; margin-bottom: 0;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
<div style="display:flex; align-items:center; gap:16px;">
<span style="color:{rank_color}; font-weight:700; font-size:2rem; font-family:monospace; line-height:1;">{original_rank_i+1:02d}</span>
<div style="display:flex; align-items:center;">
<span style="color:#FFFFFF; font-weight:600; font-size:1.35rem; letter-spacing:-0.02em;">{val["label"]}</span>
{best_badge}
</div>
</div>
<div style="text-align: right; line-height: 1.4;">
<div style="color: #E2E8F0; font-size: 0.95rem; font-family: monospace; font-weight: 500;">
<span style="color: #768390; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; font-family: sans-serif;">CV Score:</span> 
{val["cv_score"]:.4f}
</div>
<div style="color: {rank_color if is_best else '#E2E8F0'}; font-size: 0.95rem; font-family: monospace; font-weight: 600; margin-top: 4px;">
<span style="color: #768390; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; font-family: sans-serif;">{metric_name}:</span> 
{test_score_str}
</div>
</div>
</div>
<div style="margin-bottom:16px; margin-top:8px; display:flex; gap:2px; flex-wrap:wrap; align-items:center;">
{tag_badges}
</div>
<div style="background:#121824; border-radius:6px; height:8px; margin-bottom:18px; overflow:hidden;">
<div style="{bar_style}"></div>
</div>
<div class="minimal-btn-trigger {btn_class}" style="height: 20px; margin-bottom: 0;"></div>
</div>""", unsafe_allow_html=True)
                    
                    # ปุ่มที่ถูกดักแปลงรูปทรงด้วย CSS Sibling Selector ให้ขยับขึ้นไปซ้อนทับอยู่ในฐานการ์ดแบบไร้รอยต่อ
                    if st.button("คำอธิบายการทำงาน →", key=f"guide_btn_{key}", use_container_width=True):
                        show_model_explanation_dialog(val["label"])

    # ── Failed models ──
    failed = [(k, v) for k, v in competition.items() if v["cv_score"] is None]
    if failed:
        st.markdown('<div style="color:#475569;font-size:0.8rem;margin-top:8px;">โมเดลที่ train ไม่สำเร็จ: ' +
                    ", ".join(v["label"] for _, v in failed) + '</div>', unsafe_allow_html=True)


# ── Data Visualization (Insight) ───────────────────────────────────────────────

def render_viz_insight(df: pd.DataFrame, target_col: str, task_type: str):
    """Insight-focused visualization: distribution แยกตาม target class + patterns"""
    render_section_header(
        "Data Visualization",
        "Pattern และ insight ที่ซ่อนอยู่ในข้อมูล หลัง transformation",
    )

    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col]
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c != target_col]
    is_clf   = task_type == "classification"

    if not num_cols:
        st.info("ไม่มี numeric feature สำหรับวิเคราะห์")
        return

    # ── Section 1: Feature Distribution by Target ──
    st.markdown('<div style="font-size:0.8rem;color:#94A3B8;font-family:monospace;letter-spacing:0.1em;margin:16px 0 8px 0;">FEATURE DISTRIBUTION BY TARGET</div>', unsafe_allow_html=True)

    sel_feature = st.selectbox(
        "เลือก Feature",
        num_cols,
        key="viz_insight_feat",
        label_visibility="collapsed",
    )

    if is_clf and df[target_col].nunique() <= 20:
        # Box plot แยกตาม class
        fig = px.box(
            df, x=target_col, y=sel_feature,
            color=target_col,
            color_discrete_sequence=["#7AA2F7","#9ECE6A","#F7768E","#E0AF68","#BB9AF7","#7DCFFF"],
            points="outliers",
        )
        fig.update_layout(
            template="plotly_dark", height=380, showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
            xaxis_title=target_col, yaxis_title=sel_feature,
        )
        st.plotly_chart(fig, width="stretch")

        # Insight: median ต่างกันไหม
        grp = df.groupby(target_col)[sel_feature]
        medians = grp.median().sort_values()
        spread  = medians.max() - medians.min()
        rel_spread = spread / (df[sel_feature].std() + 1e-9)
        if rel_spread > 1.0:
            insight_text = f"feature นี้แยก class ได้ดีมาก — median ต่างกัน **{spread:.2f}** ({rel_spread:.1f}× std)"
            insight_color = "#9ECE6A"
        elif rel_spread > 0.3:
            insight_text = f"feature นี้แยก class ได้ปานกลาง — median ต่างกัน **{spread:.2f}**"
            insight_color = "#E0AF68"
        else:
            insight_text = f"feature นี้แยก class ได้น้อย — distribution ของแต่ละ class ทับซ้อนกันมาก"
            insight_color = "#F7768E"

        st.markdown(f"""
<div style="background:{insight_color}11;border:1px solid {insight_color}33;border-left:4px solid {insight_color};
border-radius:6px;padding:10px 14px;margin-top:4px;font-size:0.9rem;color:#E2E8F0;line-height:1.6;">
  <b style="color:{insight_color};">Insight:</b> {insight_text}<br>
  <span style="color:#94A3B8;font-size:0.82rem;">
    {"  |  ".join(f"{cls}: median={med:.2f}" for cls, med in medians.items())}
  </span>
</div>""", unsafe_allow_html=True)

    else:
        # Regression: scatter feature vs target + trend
        sample_df = df[[sel_feature, target_col]].dropna().sample(min(1000, len(df)), random_state=42)
        fig = px.scatter(
            sample_df, x=sel_feature, y=target_col,
            opacity=0.5, trendline="ols",
            color_discrete_sequence=["#7AA2F7"],
            trendline_color_override="#F7768E",
        )
        fig.update_layout(
            template="plotly_dark", height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig, width="stretch")
        corr = sample_df[[sel_feature, target_col]].corr().iloc[0, 1]
        corr_abs = abs(corr)
        strength = "สูง" if corr_abs >= 0.7 else ("ปานกลาง" if corr_abs >= 0.4 else "ต่ำ")
        direction = "บวก (ขึ้นพร้อมกัน)" if corr > 0 else "ลบ (ค่าหนึ่งขึ้น อีกค่าลง)"
        st.caption(f"Pearson r = **{corr:.3f}** — ความสัมพันธ์ {strength} เชิง{direction}")

    st.divider()

    # ── Section 2: Top Features Separation Power ──
    st.markdown('<div style="font-size:0.8rem;color:#94A3B8;font-family:monospace;letter-spacing:0.1em;margin-bottom:12px;">TOP FEATURES — SEPARATION POWER</div>', unsafe_allow_html=True)

    if is_clf and df[target_col].nunique() <= 20:
        # คำนวณ separation power = std of class medians / overall std
        sep_scores = {}
        for col in num_cols:
            try:
                class_medians = df.groupby(target_col)[col].median()
                sep = (class_medians.max() - class_medians.min()) / (df[col].std() + 1e-9)
                sep_scores[col] = round(float(sep), 3)
            except Exception:
                pass
        if sep_scores:
            sep_df = pd.DataFrame({"Feature": list(sep_scores.keys()), "Separation": list(sep_scores.values())})
            sep_df = sep_df.sort_values("Separation", ascending=False).head(15)
            fig2 = px.bar(
                sep_df, x="Separation", y="Feature", orientation="h",
                color="Separation",
                color_continuous_scale=[[0, "rgba(122,162,247,0.2)"], [1, "#9ECE6A"]],
            )
            fig2.update_layout(
                template="plotly_dark", height=max(280, len(sep_df) * 32),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                margin=dict(t=10, b=20, l=10, r=20),
                xaxis_title="Separation Score (ยิ่งสูง = แยก class ได้ดีกว่า)",
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption("Separation Score = ช่วง median ระหว่าง class ÷ std รวม  สูง = feature นี้ช่วยแยก class ได้ดี")
    else:
        # Regression: correlation กับ target
        corr_series = df[num_cols].corrwith(df[target_col]).dropna().sort_values(key=abs, ascending=False).head(15)
        corr_df = pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
        corr_df["color"] = corr_df["Correlation"].apply(lambda v: "positive" if v >= 0 else "negative")
        fig2 = px.bar(
            corr_df, x="Correlation", y="Feature", orientation="h",
            color="color", color_discrete_map={"positive": "#9ECE6A", "negative": "#F7768E"},
            range_x=[-1, 1], text=corr_df["Correlation"].round(2),
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            template="plotly_dark", height=max(280, len(corr_df) * 32),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(autorange="reversed"), showlegend=False,
            margin=dict(t=10, b=20, l=10, r=60),
            xaxis_title="Pearson r กับ Target",
        )
        st.plotly_chart(fig2, width="stretch")

    st.divider()

    # ── Section 3: Correlation Heatmap ──
    if len(num_cols) >= 2:
        st.markdown('<div style="font-size:0.8rem;color:#94A3B8;font-family:monospace;letter-spacing:0.1em;margin-bottom:12px;">FEATURE CORRELATION HEATMAP</div>', unsafe_allow_html=True)
        corr_matrix = df[num_cols].corr()
        show_text   = ".2f" if len(num_cols) <= 12 else False
        fig3 = px.imshow(corr_matrix, text_auto=show_text,
                         color_continuous_scale="RdBu_r", range_color=[-1, 1], aspect="auto")
        fig3.update_layout(
            template="plotly_dark", height=min(600, max(320, len(num_cols) * 35)),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig3, width="stretch")
        st.caption("สีแดงเข้ม = สัมพันธ์บวกสูง (+1) | สีน้ำเงินเข้ม = สัมพันธ์ลบสูง (−1) | สีจาง = ไม่สัมพันธ์กัน (≈0)")

        high_corr = [
            (corr_matrix.columns[i], corr_matrix.columns[j], round(corr_matrix.iloc[i, j], 2))
            for i in range(len(corr_matrix.columns))
            for j in range(i + 1, len(corr_matrix.columns))
            if abs(corr_matrix.iloc[i, j]) > 0.8
        ]
        if high_corr:
            pairs = " | ".join(f"`{a}` & `{b}` (r={r})" for a, b, r in high_corr[:5])
            st.warning(f"พบ Feature ที่มีความสัมพันธ์กันสูง (|r| > 0.8): {pairs} — อาจเกิด Multicollinearity กับ Linear-based models")