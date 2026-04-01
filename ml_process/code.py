"""
ml_process/code.py  —  Entry point (UI เท่านั้น)
รับข้อมูลจาก session_state["main_df"] ที่ผ่าน Data Preparation แล้ว

Flow:
  1. เลือก Target Column
  2. แสดง Data Splitting (80/20)
  3. กด Run → preprocess + competition + evaluate
  4. แสดง Leaderboard + Best Params + Metrics + Visualization
"""
import streamlit as st

from ml_process.preprocess  import preprocess, detect_task
from ml_process.runner      import run_competition
from ml_process.evaluation  import (
    get_metrics,
    show_leaderboard,
    show_metrics,
    show_confusion_matrix,
    show_pred_vs_actual,
)


def render_ml_process():
    from app import page_header

    page_header(
        "ML Process",
        "Auto Model Competition — Train ทุก model, เลือกตัวที่ดีสุด, Evaluate บน Test set",
    )

    # ── Guard: ต้องมีข้อมูลก่อน ───────────────────────────────
    if st.session_state.get("main_df") is None:
        st.warning("ไม่พบข้อมูล — กรุณากลับไปทำ Data Preparation ก่อน")
        if st.button("← กลับไป Upload"):
            st.query_params["step"] = "upload"
            st.rerun()
        return

    df        = st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")

    st.info(
        f"**Current Dataset:** {file_name}  "
        f"|  {df.shape[0]:,} rows × {df.shape[1]} columns"
    )

    # ── 1. Target Column ──────────────────────────────────────
    st.subheader("1. กำหนด Target Column")

    # ใช้ค่าที่ส่งมาจาก Transformation ถ้ามี
    preset_idx = 0
    preset_col = st.session_state.get("ml_target_col_preset")
    if preset_col and preset_col in df.columns.tolist():
        preset_idx = df.columns.tolist().index(preset_col)
    else:
        preset_idx = len(df.columns) - 1

    target_col = st.selectbox(
        "Target",
        df.columns.tolist(),
        index=preset_idx,
        key="ml_target_col",
        label_visibility="collapsed",
    )

    task_detected = detect_task(df, target_col)
    st.caption(
        f"Task ที่ตรวจพบ: **{task_detected.upper()}**  "
        f"|  Unique values ใน target: **{df[target_col].nunique()}**"
    )

    # ── 2. Data Splitting Preview ─────────────────────────────
    st.subheader("2. Data Splitting (80 / 20)")

    n_total = len(df)
    n_train = int(n_total * 0.8)
    n_test  = n_total - n_train

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",      f"{n_total:,}")
    c2.metric("Train Set (80%)", f"{n_train:,}")
    c3.metric("Test Set (20%)",  f"{n_test:,}")

    # ── 3. Run Competition ────────────────────────────────────
    st.subheader("3. Model Competition + Auto Hyperparameter Tuning")

    # ── Info box ──────────────────────────────────────────────
    st.markdown("""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;
padding:16px 20px;margin-bottom:20px">
  <div style="font-weight:600;font-size:0.95rem;color:#e6edf3;margin-bottom:10px">
    Auto Model Competition
  </div>
  <div style="font-size:0.85rem;color:#8b949e;line-height:1.9">
    1. Train <b style="color:#e6edf3">ทุก model</b> บน Train set (80%) พร้อม Auto Hyperparameter Tuning<br>
    2. เปรียบเทียบ Cross-Validation Score ของแต่ละตัว<br>
    3. เลือก <b style="color:#58a6ff">Best Model</b> อัตโนมัติ<br>
    4. Evaluate บน Test set (20%) ที่ยังไม่เคยเห็น
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Model cards ───────────────────────────────────────────
    from ml_process.runner import get_available_models
    from ml_process.config import MODELS_CLASSIFICATION, MODELS_REGRESSION
    from ml_process.config import (
        MODELS_CLASSIFICATION as _MC,
        MODELS_REGRESSION as _MR,
    )

    # คำอธิบายแต่ละ model
    _DESCS = {
        "random_forest":               "ทนทานต่อ noise ดี เหมาะเป็น anchor",
        "gradient_boosting":           "แก้ error ทีละขั้น มักให้ accuracy สูงสุด",
        "logistic_regression":         "เร็ว อธิบายได้ง่าย ช่วย balance ensemble",
        "decision_tree":               "เห็น decision boundary ชัด เพิ่ม diversity",
        "svm":                         "แข็งแกร่งกับ high-dimensional data",
        "knn":                         "ใช้ k เพื่อนบ้านที่ใกล้ที่สุดในการตัดสินใจ",
        "naive_bayes":                 "เร็วมาก เหมาะเป็น baseline",
        "xgboost":                     "Boosting ที่แม่นยำสูง ชนะ Kaggle หลายรายการ",
        "lightgbm":                    "เร็วกว่า XGBoost เหมาะกับข้อมูลใหญ่",
        "catboost":                    "จัดการ categorical ได้ดีโดยไม่ต้อง encode",
        "linear_regression":           "เร็ว stable เหมาะกับข้อมูล linear",
        "decision_tree_regressor":     "จับ non-linear ได้ เพิ่ม diversity",
        "random_forest_regressor":     "ทนทาน เหมาะเป็น anchor",
        "gradient_boosting_regressor": "accuracy สูงสุดสำหรับ regression",
        "knn_regressor":               "ใช้ k เพื่อนบ้านในการประมาณค่า",
        "xgboost_regressor":           "Boosting แม่นยำสูงสำหรับ regression",
        "lightgbm_regressor":          "เร็วกว่า XGBoost สำหรับ regression",
        "catboost_regressor":          "จัดการ categorical ได้ดีสำหรับ regression",
    }

    # ตรวจก่อนว่า task type คืออะไร (ใช้ detect_task แบบ quick)
    _task_preview = detect_task(df, target_col)
    _avail = get_available_models(_task_preview)

    st.markdown(
        f'<div style="font-weight:600;font-size:0.88rem;color:#8b949e;'
        f'margin-bottom:12px">Models ที่จะเข้าแข่งขัน ({len(_avail)} ตัว)</div>',
        unsafe_allow_html=True
    )

    # render 2 คอลัมน์
    keys = list(_avail.keys())
    col_a, col_b = st.columns(2)
    for i, key in enumerate(keys):
        label = _avail[key]
        desc  = _DESCS.get(key, "")
        card_html = f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;
padding:12px 16px;margin:4px 0;display:flex;align-items:flex-start;gap:10px">
  <div style="width:10px;height:10px;border-radius:50%;background:#3fb950;
  flex-shrink:0;margin-top:4px"></div>
  <div>
    <div style="font-family:'JetBrains Mono',monospace;font-weight:600;
    font-size:0.85rem;color:#e6edf3">{label}</div>
    <div style="font-size:0.76rem;color:#8b949e;margin-top:2px">{desc}</div>
  </div>
</div>"""
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("▶  Run All Models", type="primary"):
        st.session_state.pop("ml_result",  None)
        st.session_state.pop("ml_metrics", None)

        with st.spinner("กำลัง preprocess ข้อมูล..."):
            try:
                X_train, X_test, y_train, y_test, task_type = preprocess(df, target_col)
                st.session_state["ml_task_type"] = task_type
            except Exception as e:
                st.error(f"Preprocess ล้มเหลว: {e}")
                return

        progress_bar = st.progress(0, text="กำลัง train models...")

        def on_progress(label, i, total):
            progress_bar.progress(
                (i + 1) / total,
                text=f"Training {label}... ({i+1}/{total})"
            )

        try:
            result  = run_competition(
                X_train, X_test, y_train, y_test, task_type,
                on_progress=on_progress
            )
            metrics = get_metrics(result["y_test"], result["y_pred"], task_type)
            progress_bar.empty()

            st.session_state["ml_result"]  = result
            st.session_state["ml_metrics"] = metrics
            st.rerun()

        except Exception as e:
            progress_bar.empty()
            st.error(f"เกิดข้อผิดพลาด: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ── 4. Results ────────────────────────────────────────────
    result  = st.session_state.get("ml_result")
    metrics = st.session_state.get("ml_metrics")

    if result and metrics:
        task_type   = result["task_type"]
        best_label  = result["best_label"]
        best_params = result["best_params"]

        st.success(f"✅  Best Model: **{best_label}**")

        # ── Debug: ตรวจสอบ y_test vs y_pred ──────────────────
        with st.expander("🔍 ตรวจสอบผล Prediction (Debug)", expanded=False):
            import pandas as pd
            y_test_s = result["y_test"]
            y_pred_s = result["y_pred"]
            st.write("**Target column distribution (y_test):**")
            st.write(pd.Series(y_test_s).value_counts().head(10))
            st.write("**Prediction sample (y_pred vs y_test) — 10 แถวแรก:**")
            debug_df = pd.DataFrame({
                "y_test": list(y_test_s)[:10],
                "y_pred": list(y_pred_s)[:10],
                "correct": [str(a)==str(b) for a,b in zip(list(y_test_s)[:10], list(y_pred_s)[:10])]
            })
            st.dataframe(debug_df, use_container_width=True)
            st.write(f"**Task type detected:** `{task_type}`")

        st.subheader("4. Model Leaderboard (Cross-Validation Score)")
        show_leaderboard(result["competition"])

        if best_params:
            with st.expander(f"Best Hyperparameters ของ {best_label}"):
                for k, v in best_params.items():
                    st.write(f"- **{k}**: `{v}`")

        st.subheader("5. Evaluation — Test Set (20%)")
        st.caption(
            f"ผลลัพธ์จาก **{best_label}** predict บน Test set ที่ยังไม่เคยเห็น"
        )
        show_metrics(metrics)

        st.divider()

        if task_type == "classification":
            st.write("**Confusion Matrix**")
            show_confusion_matrix(result["y_test"], result["y_pred"])
        else:
            st.write("**Actual vs Predicted**")
            show_pred_vs_actual(result["y_test"].values, result["y_pred"])

    # ── Navigation ────────────────────────────────────────────
    st.divider()
    col1, _space, col2 = st.columns([0.8, 8, 0.8])

    with col1:
        if st.button("Back", type="secondary", width="stretch"):
            st.query_params["step"] = "transformation"
            st.rerun()

    with col2:
        if st.button(
            "Next Step", type="primary", width="stretch",
            disabled=(result is None),
        ):
            st.query_params["step"] = "explainable"
            st.rerun()
        if result is None:
            st.caption(
                "กด Run All Models ก่อนไปขั้นตอนถัดไป",
                width="content", text_alignment="center",
            )