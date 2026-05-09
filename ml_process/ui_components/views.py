"""ml_process/views.py — Streamlit / Plotly rendering (no business logic)"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

from ml_process.logic.config import MODEL_WHY


# ── Target Info ───────────────────────────────────────────────────────────────

def render_target_info(target_col: str, task: str, n_unique: int):
    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:10px 16px;margin-bottom:16px;display:flex;gap:16px;align-items:center">
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Target Column</div>
    <div style="font-family:monospace;font-weight:700;font-size:0.95rem;
    color:#58a6ff">{target_col}</div>
  </div>
  <div style="width:1px;height:32px;background:#30363d"></div>
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Task</div>
    <div style="font-weight:700;font-size:0.88rem;color:#3fb950">
      {task.upper()}
    </div>
  </div>
  <div style="width:1px;height:32px;background:#30363d"></div>
  <div>
    <div style="font-size:0.75rem;color:#8b949e;text-transform:uppercase;
    letter-spacing:1px;margin-bottom:2px">Unique Values</div>
    <div style="font-weight:700;font-size:0.88rem;color:#e6edf3">
      {n_unique}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Competition Description ───────────────────────────────────────────────────

def render_competition_desc():
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 18px;margin-bottom:16px;font-size:0.83rem;color:#8b949e;line-height:1.8">
  1. Train <b style="color:#e6edf3">ทุก model</b> บน Train set (80%) พร้อม Auto Hyperparameter Tuning<br>
  2. เปรียบเทียบ Cross-Validation Score<br>
  3. เลือก <b style="color:#58a6ff">Best Model</b> อัตโนมัติ<br>
  4. Evaluate บน Test set (20%)<br>
</div>""", unsafe_allow_html=True)


# ── Model Cards ───────────────────────────────────────────────────────────────

def render_model_cards(avail: dict, model_desc: dict):
    st.markdown(f'<div style="font-weight:600;font-size:0.85rem;color:#8b949e;margin-bottom:10px">'
                f'Models ที่จะแข่งขัน ({len(avail)} ตัว)</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, (key, label) in enumerate(avail.items()):
        card = (f'<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;'
                f'padding:10px 14px;margin:4px 0;display:flex;align-items:flex-start;gap:10px">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:#3fb950;'
                f'flex-shrink:0;margin-top:5px"></div>'
                f'<div><div style="font-family:monospace;font-weight:600;font-size:0.83rem;'
                f'color:#e6edf3">{label}</div>'
                f'<div style="font-size:0.74rem;color:#8b949e;margin-top:1px">'
                f'{model_desc.get(key, "")}</div></div></div>')
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(card, unsafe_allow_html=True)


# ── Best Model Explanation ────────────────────────────────────────────────────

def render_best_model_card(result: dict, best_label: str):
    ranked = sorted([(k, v) for k, v in result["competition"].items() if v["cv_score"]],
                    key=lambda x: x[1]["cv_score"], reverse=True)
    if not ranked:
        st.warning("ไม่มีโมเดลที่ผ่าน cross-validation — ไม่สามารถแสดง Best Model ได้")
        return
    best_cv, best_std = ranked[0][1]["cv_score"], ranked[0][1]["cv_std"]
    avg_std = sum(v["cv_std"] for _, v in ranked) / len(ranked)

    r2_desc = (f"นำหน้า {ranked[1][1]['label']} อยู่ {round(best_cv - ranked[1][1]['cv_score'], 4)} "
               f"({round((best_cv - ranked[1][1]['cv_score']) / (ranked[1][1]['cv_score'] + 1e-9) * 100, 1)}%)"
               if len(ranked) >= 2 else "เป็นโมเดลที่มีประสิทธิภาพสูงสุดในทุกการทดสอบ")
    r3_desc = ("มีความเสถียรสูงกว่าค่าเฉลี่ย" if best_std <= avg_std else "แม้มีความผันผวนแต่คะแนนยังคงเป็นอันดับหนึ่ง")
    r4_desc = MODEL_WHY.get(best_label, "สถาปัตยกรรมเหมาะสมกับรูปแบบของข้อมูลชุดนี้ที่สุด")

    st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:14px 18px;margin-top:12px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
    <span style="background:#3fb950;color:#0d1117;font-weight:700;font-size:0.85rem;
    padding:3px 12px;border-radius:12px">BEST MODEL</span>
    <span style="font-family:monospace;font-weight:700;color:#e6edf3;font-size:1.1rem">{best_label}</span>
  </div>
  <div style="font-size:0.95rem;color:#c9d1d9;line-height:2.0">
    <b style="color:#58a6ff">1. Cross-Val Score สูงสุด</b> {best_cv:.4f} {r2_desc}<br>
    <b style="color:#3fb950">2. Consistency สูงสุด</b> ±Std = {best_std:.4f} {r3_desc}<br>
    <b style="color:#d29922">3. Model Insight</b> {r4_desc}
  </div>
  <div style="margin-top:12px;padding-top:10px;border-top:1px solid #30363d;
  font-size:0.85rem;color:#8b949e">
    Cross-Validation คือการจำลองการทดสอบหลายรอบเพื่อให้มั่นใจว่าโมเดลจะทำงานได้ดีกับข้อมูลใหม่ในอนาคต
  </div>
</div>""", unsafe_allow_html=True)


# ── Metrics Explanation ───────────────────────────────────────────────────────

def render_metrics_explain(metrics: dict, task_type: str):
    if task_type == "classification":
        acc = metrics.get("Accuracy", 0)
        f1  = metrics.get("F1(Mac)", 0)
        items = [
            ("#3fb950", "Accuracy", f"ทำนายถูก {acc*100:.1f}% ของข้อมูลทั้งหมดใน Test set"),
            ("#58a6ff", "Precision", "ความแม่นยำของการทำนายในแต่ละ Class (ทายว่าเป็น A แล้วเป็น A จริงไหม)"),
            ("#d29922", "Recall", "ความสามารถในการหาตัวอย่างใน Class นั้นๆ (ในบรรดาที่เป็น A จริงๆ เราหาเจอเท่าไหร่)"),
            ("#bc8cff", "F1 Score", f"ค่าเฉลี่ยแบบ Harmonic ระหว่าง Precision และ Recall (ค่าปัจจุบัน: {f1:.4f})")
        ]
    else:
        r2, rmse = metrics.get("R² Score", 0), metrics.get("RMSE", 0)
        items = [
            ("#3fb950", "R² Score", f"ความสามารถในการอธิบายข้อมูล (0-1) ยิ่งสูงยิ่งดี โมเดลนี้อธิบายได้ {max(0, r2)*100:.1f}%"),
            ("#f85149", "RMSE", f"ค่าความผิดพลาดเฉลี่ย (หน่วยเดียวกับ Target) ยิ่งต่ำยิ่งดี ปัจจุบันผิดพลาดเฉลี่ย ±{rmse:.4f}"),
            ("#f85149", "MSE", "ค่าความผิดพลาดกำลังสอง (ให้น้ำหนักกับ Error ตัวที่ใหญ่เป็นพิเศษ)")
        ]

    rows_html = ""
    for color, label, desc in items:
        rows_html += f"""<div style="display:flex; align-items:flex-start; gap:16px; margin-bottom:14px">
<div style="background:{color}22; color:{color}; border:1px solid {color}44; font-size:0.8rem; font-weight:700; padding:4px 0; border-radius:6px; text-transform:uppercase; margin-top:2px; width:110px; text-align:center; flex-shrink:0">{label}</div>
<div style="font-size:1.0rem; color:#8b949e; line-height:1.6; padding-top:2px">{desc}</div>
</div>"""

    st.markdown(f"""<div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:24px 28px; margin:20px 0">
<div style="font-size:0.85rem; color:#e6edf3; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:20px; opacity:0.5">Metric Glossary — คำอธิบายตัวชี้วัด</div>
<div style="display:flex; flex-direction:column">
{rows_html}
</div>
</div>""", unsafe_allow_html=True)


# ── Confusion Matrix Explanation ──────────────────────────────────────────────

def render_cm_explain(y_test, y_pred):
    from sklearn.metrics import confusion_matrix as _cm
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    labels  = sorted(list(set(y_test) | set(y_pred)), key=str)
    cm_arr  = _cm(y_test, y_pred, labels=labels)
    correct   = int(np.trace(cm_arr))
    incorrect = int(cm_arr.sum() - correct)
    total     = correct + incorrect
    
    items = [
        ("#58a6ff", "Diagonal", f"แนวทแยง = <b>ทำนายถูก</b> ({correct}/{total} = {correct/total*100:.1f}%)"),
        ("#d29922", "Outside", f"นอกแนวทแยง = <b>ทำนายผิด</b> ({incorrect} samples)"),
        ("#8b949e", "Axis", "แถว = ค่าจริง (Actual), คอลัมน์ = ที่โมเดลทาย (Predicted)")
    ]

    rows_html = ""
    for color, label, desc in items:
        rows_html += f"""<div style="display:flex; align-items:flex-start; gap:16px; margin-bottom:14px">
<div style="background:{color}22; color:{color}; border:1px solid {color}44; font-size:0.8rem; font-weight:700; padding:4px 0; border-radius:6px; text-transform:uppercase; margin-top:2px; width:110px; text-align:center; flex-shrink:0">{label}</div>
<div style="font-size:1.0rem; color:#8b949e; line-height:1.6; padding-top:2px">{desc}</div>
</div>"""

    st.markdown(f"""<div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:24px 28px; margin:20px 0">
<div style="font-size:0.85rem; color:#e6edf3; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:20px; opacity:0.5">Confusion Matrix Guide — วิธีอ่านตารางการทำนาย</div>
<div style="display:flex; flex-direction:column">
{rows_html}
</div>
</div>""", unsafe_allow_html=True)


# ── Scatter Explanation ───────────────────────────────────────────────────────

def render_scatter_explain():
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:8px 0;font-size:0.81rem;color:#c9d1d9;line-height:1.8">
  <b style="color:#e6edf3">วิธีอ่าน Actual vs Predicted</b><br>
  • <b style="color:#58a6ff">แกน X</b> = ค่าจริง, <b style="color:#58a6ff">แกน Y</b> = ค่าที่ทำนาย<br>
  • <b style="color:#3fb950">เส้นประแดง</b> = Perfect line — จุดที่ดีควรอยู่บนหรือใกล้เส้นนี้<br>
  • จุดที่กระจายสม่ำเสมอรอบเส้น = model ไม่มี systematic bias
</div>""", unsafe_allow_html=True)


# ── Feature Importance Chart ──────────────────────────────────────────────────

def render_fi(fi_df: pd.DataFrame | None, best_label: str, fi_error: str | None):
    st.caption("features ไหนมีผลต่อการตัดสินใจของ model มากที่สุด")

    if fi_df is not None:
        top_n  = min(20, len(fi_df))
        fig_fi = px.bar(fi_df.head(top_n), x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues",
                        text=fi_df.head(top_n)["Importance"].round(4))
        fig_fi.update_layout(template="plotly_dark", height=max(350, top_n * 28),
                             yaxis=dict(autorange="reversed"), coloraxis_showscale=False,
                             margin=dict(t=20, b=20))
        fig_fi.update_traces(textposition="outside")
        st.plotly_chart(fig_fi, width="stretch")

        total  = fi_df["Importance"].sum()
        medals = ["#1", "#2", "#3"]
        html = ('<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:8px 0">'
                '<div style="font-size:0.81rem;color:#c9d1d9;line-height:1.8">'
                '• ยิ่งแท่งยาว = feature มีผลต่อ model มากกว่า<br>'
                '• feature ที่ importance ต่ำมากอาจพิจารณาตัดออกเพื่อลด complexity</div>'
                '<div style="margin-top:8px">')
        for i, row in fi_df.head(3).iterrows():
            pct = row["Importance"] / (total + 1e-9) * 100
            html += (f'<div style="display:flex;align-items:center;gap:8px;padding:4px 8px;'
                     f'background:#0d1117;border-radius:4px;margin:2px 0">'
                     f'<span>{medals[i]}</span>'
                     f'<span style="font-family:monospace;color:#e6edf3;flex:1">{row["Feature"]}</span>'
                     f'<span style="color:#58a6ff">{row["Importance"]:.4f}</span>'
                     f'<span style="color:#8b949e;font-size:0.76rem">({pct:.1f}%)</span></div>')
        html += '</div></div>'
        st.markdown(html, unsafe_allow_html=True)

    elif fi_error:
        st.warning(f"คำนวณไม่ได้: {fi_error}")
    else:
        st.info(f"{best_label} ไม่รองรับ Feature Importance โดยตรง")


# ── Data Visualization Tab ────────────────────────────────────────────────────

def render_viz(df: pd.DataFrame):
    st.caption(f"ข้อมูลปัจจุบัน {df.shape[0]:,} rows × {df.shape[1]} columns (หลัง transformation)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_cols = num_cols + cat_cols
    if not all_cols:
        st.info("ไม่มีคอลัมน์สำหรับ visualize")
        return

    v1, v2 = st.columns(2)
    with v1:
        sel_col = st.selectbox("เลือกคอลัมน์", all_cols, key="viz_col_ml")
    with v2:
        color_col = st.selectbox("Color by", [None] + cat_cols, key="viz_color_ml") if cat_cols else None

    if sel_col in num_cols:
        fig = px.histogram(df, x=sel_col, color=color_col, marginal="box", nbins=30,
                           color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        counts = df[sel_col].value_counts().reset_index()
        counts.columns = [sel_col, "count"]
        fig = px.bar(counts.head(20), x=sel_col, y="count", color_discrete_sequence=["#58a6ff"])
    fig.update_layout(template="plotly_dark", height=360, showlegend=bool(color_col),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

    if len(num_cols) >= 2:
        st.divider()
        fig2 = px.imshow(df[num_cols].corr(), text_auto=".2f",
                         color_continuous_scale="RdBu_r", range_color=[-1, 1], aspect="auto")
        fig2.update_layout(template="plotly_dark", height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig2, width="stretch")


def render_metric_cards(metrics_dict: dict):
    """แสดง metrics ในรูปแบบพรีเมียมการ์ด"""
    # กรองเฉพาะค่าที่เป็นตัวเลข
    display_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
    cols = st.columns(len(display_metrics))
    
    style_map = {
        "Accuracy":       {"color": "#3fb950"},
        "Precision(Mac)": {"color": "#58a6ff"},
        "Recall(Mac)":    {"color": "#d29922"},
        "F1(Mac)":        {"color": "#bc8cff"},
        "MSE":            {"color": "#f85149"},
        "RMSE":           {"color": "#f85149"},
        "R² Score":       {"color": "#3fb950"},
    }

    for i, (name, val) in enumerate(display_metrics.items()):
        style = style_map.get(name, {"color": "#8b949e"})
        formatted_val = f"{val:.4f}" if isinstance(val, (float, np.float64, np.float32)) else str(val)
        with cols[i]:
            st.markdown(f"""
            <div style="border-bottom: 2px solid {style['color']}33; padding: 12px 0; margin-bottom: 15px">
                <div style="color:#8b949e; font-size:0.85rem; font-weight:600; text-transform:uppercase; 
                display:flex; align-items:center; gap:8px; letter-spacing:0.07em">
                    <span style="color:{style['color']}; font-size:14px">●</span> {name}
                </div>
                <div style="color:#e6edf3; font-size:2.2rem; font-weight:800; margin-top:4px; letter-spacing:-0.02em">
                    {formatted_val}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── Navigation ────────────────────────────────────────────────────────────────

def render_nav(result):
    st.divider()
    col1, _, col2 = st.columns([1.2, 7.6, 1.2])
    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            from app import navigate
            for _k in ["ml_result", "ml_metrics", "_fi_data", "ml_task_type",
                       "_ml_scaling_used", "_ml_leakage_warnings"]:
                st.session_state.pop(_k, None)
            navigate("transformation")
    with col2:
        if st.button("Next Step", type="primary", width="stretch", disabled=(result is None)):
            from app import navigate
            navigate("explainable")
        if result is None:
            st.caption("กด Run All Models ก่อนไปขั้นตอนถัดไป",
                       width="content", text_alignment="center")