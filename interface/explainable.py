import streamlit as st
import plotly.graph_objects as go

from explainable.features.explainer import get_fitted_model, compute_permutation_importance
from explainable.features.trace_log import get_log
from ml_process.features.export import build_leaderboard_df, build_predictions_df, build_html_report

# ── Design tokens ─────────────────────────────────────────────────────────────

_BG       = "#161b22"
_BORDER   = "#30363d"
_TEXT     = "#c9d1d9"
_TEXT_DIM = "#8b949e"
_R        = "10px"
_PAD      = "16px 20px"
_GAP      = "margin: 12px 0"


def _section_header(title: str, subtitle: str = "") -> None:
    sub = (
        f'<div style="color:{_TEXT_DIM};font-size:0.9rem;margin-top:4px">{subtitle}</div>'
        if subtitle else ""
    )
    st.markdown(
        f'<div style="{_GAP}">'
        f'<div style="color:#e6edf3;font-size:1.15rem;font-weight:700">{title}</div>'
        f'{sub}</div>',
        unsafe_allow_html=True,
    )


# ── Tab 1 — Feature Importance ────────────────────────────────────────────────

def _render_importance(model, X_test, y_test, task_type):
    _section_header(
        "Feature ไหนสำคัญที่สุด?",
        "ระบบสลับค่าของแต่ละ feature แล้วดูว่า model แย่ลงแค่ไหน — แท่งยาว = สำคัญมาก",
    )

    cache_key = f"_xai_perm_{id(model)}"
    if st.session_state.get(cache_key) is None:
        with st.spinner("กำลังวิเคราะห์ความสำคัญของ feature..."):
            perm_df = compute_permutation_importance(model, X_test, y_test, task_type)
            st.session_state[cache_key] = perm_df
    perm_df = st.session_state[cache_key]

    top_n   = min(15, len(perm_df))
    plot_df = perm_df.head(top_n)

    fig = go.Figure(go.Bar(
        x=plot_df["Importance"],
        y=plot_df["Feature"],
        orientation="h",
        marker=dict(color=plot_df["Importance"].tolist(), colorscale="Blues", showscale=False),
        text=plot_df["Importance"].round(3).tolist(),
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, top_n * 36),
        yaxis=dict(autorange="reversed"),
        xaxis_title="ความสำคัญ (ยิ่งสูง = ยิ่งสำคัญ)",
        margin=dict(t=10, b=30, l=10, r=90),
    )
    st.plotly_chart(fig, width="stretch")

    positive = perm_df[perm_df["Importance"] > 0]
    if len(positive) == 0:
        st.warning("ทุก feature มีความสำคัญน้อยมาก — ลองตรวจสอบ dataset อีกครั้ง")
        return

    _section_header("3 Feature สำคัญที่สุด")

    rank_color  = ["#58a6ff", "#3fb950", "#d29922"]
    rank_label  = ["#1", "#2", "#3"]
    total       = positive["Importance"].sum() + 1e-9
    top3        = positive.head(3).reset_index(drop=True)
    cols        = st.columns(len(top3))

    for i, col in enumerate(cols):
        row = top3.iloc[i]
        pct = row["Importance"] / total * 100
        with col:
            st.markdown(
                f'<div style="background:{_BG};border:1px solid {_BORDER};'
                f'border-top:3px solid {rank_color[i]};border-radius:{_R};'
                f'padding:20px 16px;text-align:center">'
                f'<div style="color:{rank_color[i]};font-size:0.85rem;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:10px">{rank_label[i]}</div>'
                f'<div style="font-family:monospace;color:#e6edf3;font-size:1rem;'
                f'word-break:break-all;margin-bottom:12px">{row["Feature"]}</div>'
                f'<div style="color:{rank_color[i]};font-size:1.6rem;font-weight:800;'
                f'line-height:1">{pct:.0f}%</div>'
                f'<div style="color:{_TEXT_DIM};font-size:0.85rem;margin-top:6px">'
                f'ของความสำคัญรวม</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    useless = perm_df[perm_df["Importance"] <= 0]
    if len(useless):
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            f"Feature ที่แทบไม่มีผล ({len(useless)} ตัว): "
            + ", ".join(f"`{f}`" for f in useless["Feature"].head(5))
            + (" ..." if len(useless) > 5 else "")
        )


# ── Tab 2 — Model Guide ───────────────────────────────────────────────────────

_MODEL_GUIDE = {
    "Random Forest": {
        "simple":   "สร้าง Decision Tree หลายต้นพร้อมกัน แล้วให้ทุกต้น vote ผลลัพธ์ เหมือนถามหลายคนแล้วใช้เสียงส่วนใหญ่",
        "strength": "ทนทานต่อข้อมูลที่ผิดปกติ ใช้งานได้ดีกับข้อมูลหลายประเภทโดยไม่ต้อง scale",
        "weakness": "ตีความยากกว่า Decision Tree เดี่ยว ใช้ memory มากขึ้น",
    },
    "Gradient Boosting": {
        "simple":   "สร้าง tree ทีละต้น แต่ละต้นพยายามแก้ข้อผิดพลาดของต้นก่อนหน้า — เรียนรู้จากความผิดพลาดไปเรื่อยๆ",
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
        "simple":   "หาสูตรเส้นตรงที่ fit ข้อมูลได้ดีที่สุด — y = w₁×x₁ + w₂×x₂ + ... + c",
        "strength": "ตีความง่ายที่สุด เร็วมาก เห็น coefficient ของแต่ละ feature ทันที",
        "weakness": "จับ pattern โค้งหรือซับซ้อนไม่ได้",
    },
    "SVM": {
        "simple":   "หาเส้นตรง (hyperplane) ที่แบ่ง class ออกจากกัน train ด้วย SGD ทำให้เร็วมากแม้ข้อมูลใหญ่",
        "strength": "เร็วมาก รองรับข้อมูลขนาดใหญ่ได้ดี เหมาะกับ high-dimensional data",
        "weakness": "จับ pattern ที่ไม่เป็นเส้นตรงไม่ได้ ต้องการ feature scaling",
    },
    "kNN": {
        "simple":   "ทำนายโดยดูจาก k ตัวอย่างที่ใกล้เคียงที่สุด — ถ้าเพื่อนบ้าน k คนส่วนใหญ่เป็น class A ก็ทำนายว่าเป็น class A",
        "strength": "ง่ายมาก ไม่ต้อง train",
        "weakness": "ช้ากับข้อมูลใหญ่ ต้องการ scaling",
    },
    "Naive Bayes": {
        "simple":   "คำนวณความน่าจะเป็นว่าข้อมูลนี้อยู่ใน class ไหน โดยสมมติว่าแต่ละ feature กระจายแบบ Normal และไม่ขึ้นต่อกัน",
        "strength": "เร็วมากที่สุด เหมาะเป็น baseline สำหรับข้อมูลตัวเลขที่กระจายใกล้เคียง Normal",
        "weakness": "สมมติว่า feature อิสระกัน ซึ่งมักไม่จริงในทางปฏิบัติ ทำให้แม่นยำน้อยลงกับข้อมูลที่มี feature สัมพันธ์กัน",
    },
}

_METRIC_EXPLAIN = {
    "classification": [
        ("Accuracy",        "#58a6ff", "% ที่ทำนายถูกจากทั้งหมด — ใช้ได้ดีเมื่อ class ไม่เสียสมดุล"),
        ("F1(Mac)",         "#3fb950", "ค่าเฉลี่ยของ F1 ทุก class — เหมาะกับ class ที่ไม่สมดุล"),
        ("Precision(Mac)",  "#d29922", "บรรดาที่ทำนายว่าเป็น class X — มีกี่ % ที่ถูกจริง"),
        ("Recall(Mac)",     "#bc8cff", "บรรดาที่เป็น class X จริงๆ — model ตรวจพบได้กี่ %"),
    ],
    "regression": [
        ("R² Score", "#58a6ff", "model อธิบาย variance ของข้อมูลได้กี่ % (1.0 = perfect)"),
        ("RMSE",     "#3fb950", "error เฉลี่ยในหน่วยเดียวกับ target — ยิ่งต่ำยิ่งดี"),
        ("MSE",      "#d29922", "เหมือน RMSE แต่ยกกำลัง 2 — error ใหญ่ถูก penalize มากกว่า"),
    ],
}

_CARD_COLS = [
    ("#58a6ff", "หลักการ",  "simple"),
    ("#3fb950", "จุดแข็ง",  "strength"),
    ("#f85149", "จุดอ่อน",  "weakness"),
]


def _render_guide(best_label: str, task_type: str, metrics: dict):
    guide = None
    for key, val in _MODEL_GUIDE.items():
        if key.lower() in best_label.lower() or best_label.lower() in key.lower():
            guide = val
            break

    if guide:
        _section_header(
            f"{best_label} ทำงานอย่างไร?",
            "ทำความเข้าใจ model ที่ระบบเลือกให้",
        )
        cards_html = "".join(
            f'<div style="flex:1;background:{_BG};border:1px solid {_BORDER};'
            f'border-top:3px solid {color};border-radius:{_R};padding:{_PAD}">'
            f'<div style="color:{color};font-weight:700;font-size:0.9rem;'
            f'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px">{label}</div>'
            f'<div style="color:{_TEXT};font-size:1rem;line-height:1.8">{guide[field]}</div>'
            f'</div>'
            for color, label, field in _CARD_COLS
        )
        st.markdown(
            f'<div style="display:flex;gap:16px;align-items:stretch">{cards_html}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    metric_rows = _METRIC_EXPLAIN.get(task_type, [])
    _section_header(
        "ผลลัพธ์บน Test Set",
        "ข้อมูลที่ model ไม่เคยเห็นตอน train — สะท้อนประสิทธิภาพจริง",
    )

    val_cols = st.columns(len(metric_rows))
    for col, (name, color, _) in zip(val_cols, metric_rows):
        val = metrics.get(name, "—")
        with col:
            st.markdown(
                f'<div style="background:{_BG};border:1px solid {_BORDER};'
                f'border-radius:{_R};padding:{_PAD};text-align:center">'
                f'<div style="color:{_TEXT_DIM};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.05em;text-transform:uppercase;margin-bottom:10px">{name}</div>'
                f'<div style="color:{color};font-size:1.8rem;font-weight:800;line-height:1">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    _section_header("ความหมายของแต่ละ Metric")

    for name, color, explain in metric_rows:
        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_BORDER};'
            f'border-left:3px solid {color};border-radius:0 {_R} {_R} 0;'
            f'padding:{_PAD};{_GAP};display:flex;gap:16px;align-items:flex-start">'
            f'<div style="font-family:monospace;font-weight:700;color:{color};'
            f'min-width:108px;font-size:1rem;flex-shrink:0">{name}</div>'
            f'<div style="color:{_TEXT};font-size:1rem;line-height:1.75">{explain}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Tab 3 — Pipeline Trace ────────────────────────────────────────────────────

_STEP_COLOR = {
    "Upload":              "#58a6ff",
    "Data Cleaning":       "#3fb950",
    "Data Transformation": "#d29922",
    "ML Process":          "#bc8cff",
}
_STEP_ICON = {
    "Upload":              "📂",
    "Data Cleaning":       "🧹",
    "Data Transformation": "⚙️",
    "ML Process":          "🏆",
}
_STEP_ORDER = ["Upload", "Data Cleaning", "Data Transformation", "ML Process"]


def _render_trace():
    _section_header(
        "บันทึกการตัดสินใจตลอด Pipeline",
        "ทุกขั้นตอนตั้งแต่ Upload ถึง ML Process — พร้อมเหตุผลว่า \"ทำไม\" ระบบจึงตัดสินใจแบบนั้น",
    )

    log = get_log()
    if not log:
        st.info("ยังไม่มีข้อมูล — ต้องเริ่ม pipeline ตั้งแต่ขั้นตอน Upload ใหม่")
        return

    # ── Step progress bar ──────────────────────────────────────────────────────
    completed = {e.get("step") for e in log}
    n_done = sum(1 for s in _STEP_ORDER if s in completed)

    # progress bar
    pct = int(n_done / len(_STEP_ORDER) * 100)
    st.markdown(
        f'<div style="background:{_BORDER};border-radius:6px;height:6px;margin-bottom:8px">'
        f'<div style="background:linear-gradient(90deg,#58a6ff,#bc8cff);width:{pct}%;'
        f'height:6px;border-radius:6px;transition:width 0.3s"></div></div>',
        unsafe_allow_html=True,
    )

    prog_cols = st.columns(len(_STEP_ORDER))
    for col, step in zip(prog_cols, _STEP_ORDER):
        color   = _STEP_COLOR[step]
        icon    = _STEP_ICON[step]
        done    = step in completed
        opacity = "1" if done else "0.28"
        check   = "✓" if done else ""
        with col:
            st.markdown(
                f'<div style="text-align:center;opacity:{opacity};padding:4px 0">'
                f'<div style="font-size:1.4rem;margin-bottom:4px">{icon}</div>'
                f'<div style="color:{color};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.03em">{step}</div>'
                f'<div style="color:{color};font-size:0.7rem;margin-top:2px">{check}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Timeline cards ─────────────────────────────────────────────────────────
    for idx, entry in enumerate(log):
        step  = entry.get("step", "")
        items = entry.get("items", [])
        explanations = entry.get("explanations", [])
        color = _STEP_COLOR.get(step, "#8b949e")
        icon  = _STEP_ICON.get(step, "📋")

        header_items = [it for it in items if not it.startswith("  ")]
        detail_items = [it for it in items if it.startswith("  ")]

        # ── WHAT happened rows ─────────────────────────────────────────────
        rows_html = ""
        for it in header_items:
            if it.endswith(":"):
                rows_html += (
                    f'<div style="color:{color};font-size:0.85rem;font-weight:700;'
                    f'letter-spacing:0.04em;text-transform:uppercase;padding:10px 0 4px">{it}</div>'
                )
            else:
                rows_html += (
                    f'<div style="color:{_TEXT};font-size:0.95rem;padding:4px 0;'
                    f'border-bottom:1px solid rgba(48,54,61,0.4)">{it}</div>'
                )

        # ── Detail items (expandable) ──────────────────────────────────────
        detail_html = "".join(
            f'<div style="color:{_TEXT_DIM};font-size:0.88rem;padding:3px 0">'
            f'<code style="color:{_TEXT};background:rgba(255,255,255,0.04);'
            f'border-radius:4px;padding:1px 6px">{it.strip()}</code></div>'
            for it in detail_items
        )

        detail_section = ""
        if detail_items:
            detail_section = (
                f'<details style="margin-top:10px">'
                f'<summary style="cursor:pointer;color:{_TEXT_DIM};font-size:0.88rem;'
                f'font-weight:600;user-select:none">'
                f'ดูรายละเอียด ({len(detail_items)} รายการ)</summary>'
                f'<div style="padding:8px 0 0 4px">{detail_html}</div>'
                f'</details>'
            )

        # ── WHY explanations ───────────────────────────────────────────────
        why_section = ""
        if explanations:
            why_items_html = ""
            for exp in explanations:
                # Warning items
                if exp.startswith("⚠"):
                    why_items_html += (
                        f'<div style="color:#d29922;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(210,153,34,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{exp}</div>'
                    )
                # Success items
                elif exp.startswith("✓"):
                    why_items_html += (
                        f'<div style="color:#3fb950;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(63,185,80,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{exp}</div>'
                    )
                # Info items
                elif exp.startswith("ℹ"):
                    why_items_html += (
                        f'<div style="color:#8b949e;font-size:0.9rem;padding:6px 10px;'
                        f'background:rgba(139,148,158,0.08);border-radius:6px;'
                        f'margin:4px 0;line-height:1.6">{exp}</div>'
                    )
                # Question headers (ทำไม...?)
                elif exp.startswith("ทำไม"):
                    why_items_html += (
                        f'<div style="color:{_TEXT};font-size:0.92rem;padding:6px 10px;'
                        f'background:rgba(88,166,255,0.06);border-left:2px solid {color};'
                        f'border-radius:0 6px 6px 0;'
                        f'margin:4px 0;line-height:1.7">{exp}</div>'
                    )
                # Sub-items (  → ...)
                elif exp.startswith("  →"):
                    why_items_html += (
                        f'<div style="color:{_TEXT_DIM};font-size:0.88rem;'
                        f'padding:2px 10px 2px 20px;line-height:1.5">{exp}</div>'
                    )
                # General explanations
                else:
                    why_items_html += (
                        f'<div style="color:{_TEXT};font-size:0.9rem;padding:5px 10px;'
                        f'margin:3px 0;line-height:1.6">{exp}</div>'
                    )

            why_section = (
                f'<div style="margin-top:14px;padding-top:14px;'
                f'border-top:1px dashed rgba(48,54,61,0.6)">'
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">'
                f'<span style="font-size:1rem">💡</span>'
                f'<span style="color:{color};font-weight:700;font-size:0.88rem;'
                f'letter-spacing:0.05em;text-transform:uppercase">ทำไม?</span></div>'
                f'{why_items_html}'
                f'</div>'
            )

        # ── Assemble card ──────────────────────────────────────────────────
        st.markdown(
            f'<div style="background:{_BG};border:1px solid {_BORDER};'
            f'border-left:3px solid {color};border-radius:0 {_R} {_R} 0;'
            f'padding:{_PAD};{_GAP}">'
            # Header with icon + step name
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">'
            f'<span style="font-size:1.3rem">{icon}</span>'
            f'<span style="color:{color};font-weight:700;font-size:0.95rem;'
            f'letter-spacing:0.04em;text-transform:uppercase">{step}</span></div>'
            # WHAT
            f'{rows_html}{detail_section}'
            # WHY
            f'{why_section}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def render_explainable():
    from app import page_header, navigate

    page_header(
        "Explain & Insight",
        "ทำความเข้าใจว่า Model ตัดสินใจอย่างไร และสรุปสิ่งที่ทำตลอด Pipeline",
    )

    result = st.session_state.get("ml_result")
    if result is None:
        st.warning("ไม่พบผล ML — กรุณา Run Model ก่อน")
        if st.button("กลับ ML Process"):
            navigate("ml_process")
        return

    df            = st.session_state.get("main_df")
    trans_summary = st.session_state.get("trans_summary", {})
    target_col    = (st.session_state.get("_trans_target_saved") or
                     st.session_state.get("target_col"))
    best_key      = result["best_key"]
    best_label    = result["best_label"]
    best_params   = result["best_params"]
    task_type     = result["task_type"]
    metrics       = st.session_state.get("ml_metrics", {})

    # ── Summary banner ────────────────────────────────────────────────────────
    banner_items = [
        ("Best Model", best_label,                          "#58a6ff"),
        ("Task",       task_type.upper(),                   "#3fb950"),
        ("Dataset",    f"{df.shape[0]:,} × {df.shape[1]}", "#d29922"),
    ]
    b_cols = st.columns(3)
    for col, (label, val, color) in zip(b_cols, banner_items):
        with col:
            st.markdown(
                f'<div style="background:{_BG};border:1px solid {_BORDER};'
                f'border-radius:{_R};padding:{_PAD};text-align:center">'
                f'<div style="color:{_TEXT_DIM};font-size:0.85rem;font-weight:600;'
                f'letter-spacing:0.07em;text-transform:uppercase;margin-bottom:8px">{label}</div>'
                f'<div style="color:{color};font-size:1.25rem;font-weight:800">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fit model (cached) ────────────────────────────────────────────────────
    dataset_id = st.session_state.get("last_uploaded_file", "")
    cache_key = f"_xai_cache_{best_key}_{hash(str(sorted(best_params.items())))}_{dataset_id}"
    if st.session_state.get("_xai_cache_id") != cache_key:
        with st.spinner(f"กำลัง train {best_label} เพื่อวิเคราะห์..."):
            try:
                model, X_train, X_test, _, y_test, _ = get_fitted_model(
                    df, target_col, best_key, best_params, trans_summary
                )
                st.session_state["_xai_model"]    = model
                st.session_state["_xai_X_train"]  = X_train
                st.session_state["_xai_X_test"]   = X_test
                st.session_state["_xai_y_test"]   = y_test
                st.session_state["_xai_cache_id"] = cache_key
                for k in list(st.session_state.keys()):
                    if k.startswith("_xai_perm_"):
                        del st.session_state[k]
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    model  = st.session_state["_xai_model"]
    X_test = st.session_state["_xai_X_test"]
    y_test = st.session_state["_xai_y_test"]

    from explainable.features.pipeline_state import has_comparison, get_comparison, clear_comparison, STEP_LABELS

    def _render_comparison():
        comparison = get_comparison()
        if not comparison:
            st.info("ยังไม่มีข้อมูลการเปรียบเทียบ (ข้อมูลจะปรากฏขึ้นหากคุณกดย้อนกลับไปแก้ไขขั้นตอนก่อนหน้าแล้วทำใหม่จนเสร็จ)")
            return

        _section_header(
            "Pipeline Comparison (Split View)",
            "เปรียบเทียบความแตกต่างแบบฝั่งซ้าย (เดิม) และฝั่งขวา (ใหม่)",
        )

        def flatten_keys(d, prefix=""):
            items = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    items.update(flatten_keys(v, prefix=f"{k}_"))
                else:
                    items[prefix + k] = v
            return items

        for step_key in ["upload", "cleaning", "transformation", "ml_process"]:
            if step_key not in comparison:
                continue
                
            data = comparison[step_key]
            prev = data.get("prev", {})
            curr = data.get("curr", {})
            
            flat_prev = flatten_keys(prev)
            flat_curr = flatten_keys(curr)
            all_keys = sorted(list(set(flat_prev.keys()) | set(flat_curr.keys())))

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
            
            diff_rows = []
            for k in all_keys:
                v_prev = flat_prev.get(k, "—")
                v_curr = flat_curr.get(k, "—")
                is_changed = str(v_prev) != str(v_curr)
                
                left_bg  = "background:rgba(248, 81, 73, 0.12);" if is_changed else "background:transparent;"
                right_bg = "background:rgba(63, 185, 80, 0.15);" if is_changed else "background:transparent;"
                
                left_color  = "#F85149" if is_changed else "#8B949E"
                right_color = "#3FB950" if is_changed else "#C9D1D9"
                
                diff_rows.append(f"""
                <div style="display:grid; grid-template-columns: 1fr 1fr; font-family:monospace; font-size:0.82rem; border-bottom:1px solid #21262D;">
                    <div style="{left_bg} color:{left_color}; padding:6px 12px; border-right:1px solid #30363D; word-break:break-all;">
                        <span style="opacity:0.5; margin-right:8px;">{"-" if is_changed else " "}</span>{k}: {v_prev}
                    </div>
                    <div style="{right_bg} color:{right_color}; padding:6px 12px; word-break:break-all;">
                        <span style="opacity:0.5; margin-right:8px;">{"+" if is_changed else " "}</span>{k}: {v_curr}
                    </div>
                </div>
                """)
            
            st.markdown("".join(diff_rows), unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("ล้างประวัติการเปรียบเทียบ", use_container_width=True):
            clear_comparison()
            st.rerun()

    tabs = ["Feature Importance", "Model Guide", "Pipeline Trace"]
    if has_comparison():
        tabs.append("Diff Views")

    tabs_obj = st.tabs(tabs)
    
    with tabs_obj[0]:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_importance(model, X_test, y_test, task_type)

    with tabs_obj[1]:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_guide(best_label, task_type, metrics)

    with tabs_obj[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        _render_trace()
        
    if has_comparison():
        with tabs_obj[3]:
            st.markdown("<br>", unsafe_allow_html=True)
            _render_comparison()


    st.divider()

    # ── Export + Nav ──────────────────────────────────────────────────────────
    import datetime as _dt, io as _io, zipfile as _zf
    from data_prepare.features.loading_data import delete_local

    fi_df_export = st.session_state.get("_fi_data", {}).get("fi_df")
    ts        = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = best_label.replace(" ", "_")

    html_report = build_html_report(result, metrics, fi_df_export).encode("utf-8")

    buf = _io.BytesIO()
    with _zf.ZipFile(buf, "w", _zf.ZIP_DEFLATED) as zf:
        zf.writestr("leaderboard.csv", build_leaderboard_df(result["competition"]).to_csv(index=False))
        zf.writestr("predictions.csv", build_predictions_df(result["y_test"], result["y_pred"], task_type).to_csv(index=False))
        zf.writestr("metrics.csv",     "\n".join(["Metric,Value"] + [f"{k},{v}" for k, v in metrics.items()]))
        if fi_df_export is not None:
            zf.writestr("feature_importance.csv", fi_df_export.to_csv(index=False))

    col_back, _sp, col_html, col_zip, col_finish = st.columns([0.8, 3.5, 1.2, 1.2, 0.8])
    with col_back:
        if st.button("Back", type="secondary", width="stretch"):
            navigate("ml_process")
    with col_html:
        st.download_button(
            "HTML Report",
            html_report,
            file_name=f"ml_report_{safe_name}_{ts}.html",
            mime="text/html",
            width="stretch",
        )
    with col_zip:
        st.download_button(
            "All CSV (ZIP)",
            buf.getvalue(),
            file_name=f"ml_results_{safe_name}_{ts}.zip",
            mime="application/zip",
            width="stretch",
        )
    with col_finish:
        @st.dialog("จบ Pipeline?")
        def confirm_finish():
            st.markdown(
                f"ผลลัพธ์ทั้งหมดของ **{best_label}** จะถูกล้างออก\n\n"
                "คุณต้องการจบและกลับไปหน้า Upload เพื่อเริ่มใหม่หรือไม่?"
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("ยืนยัน", type="primary", width="stretch"):
                    delete_local()
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    navigate("upload")
            with c2:
                if st.button("ยกเลิก", width="stretch"):
                    st.rerun()

        if st.button("Finish", type="primary", width="stretch"):
            confirm_finish()
