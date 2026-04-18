"""
explainable/features/trace_log.py
Pipeline trace log — สะสม decisions ตลอด pipeline แล้วแสดงที่หน้า Explainable
"""
import streamlit as st

_LOG_KEY     = "_trace_log"
_ACTIONS_KEY = "_cleaning_actions"   # accumulate cleaning before Confirm


# ── Core ──────────────────────────────────────────────────────────────────────

def append(entry: dict):
    """
    เพิ่ม log entry หนึ่งรายการ
    entry format: {step, icon, items: list[str], type: "success"|"info"|"warning"}
    """
    st.session_state.setdefault(_LOG_KEY, []).append(entry)


def get_log() -> list[dict]:
    return st.session_state.get(_LOG_KEY, [])


def clear():
    """เรียกเมื่อ upload ไฟล์ใหม่ — reset log ทั้งหมด"""
    st.session_state[_LOG_KEY]     = []
    st.session_state[_ACTIONS_KEY] = []


# ── Upload ────────────────────────────────────────────────────────────────────

def log_upload(df, file_name: str, target_col: str, task_hint: str):
    n_numeric = df.select_dtypes(include="number").shape[1]
    n_categ   = df.select_dtypes(include=["object", "category"]).shape[1]
    append({
        "step":  "Upload",
        "icon":  "",
        "type":  "success",
        "items": [
            f"File: {file_name}",
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"Numeric: {n_numeric} cols  |  Categorical: {n_categ} cols",
            f"Missing values: {int(df.isnull().sum().sum()):,} cells",
            f"Target column: {target_col}  →  {task_hint}",
        ],
    })


# ── Cleaning accumulator ──────────────────────────────────────────────────────

def track_cleaning(action_type: str, col: str, detail: str):
    """
    สะสม cleaning action ก่อน Confirm & Save
    action_type: "missing" | "outlier" | "drop_col" | "drop_dup"
    """
    actions = st.session_state.setdefault(_ACTIONS_KEY, [])
    # replace ถ้ามี entry เดิมสำหรับ col เดิม + type เดิม
    actions = [a for a in actions if not (a["col"] == col and a["type"] == action_type)]
    actions.append({"type": action_type, "col": col, "detail": detail})
    st.session_state[_ACTIONS_KEY] = actions


def track_cleaning_bulk(action_type: str, cols: list, detail: str):
    for col in cols:
        track_cleaning(action_type, col, detail)


def commit_cleaning(df_before, df_after):
    """เรียกบน Confirm & Save — รวม accumulated actions เป็น 1 log entry"""
    actions   = st.session_state.get(_ACTIONS_KEY, [])
    row_delta = df_after.shape[0] - df_before.shape[0]
    col_delta = df_after.shape[1] - df_before.shape[1]
    miss_before = int(df_before.isnull().sum().sum())
    miss_after  = int(df_after.isnull().sum().sum())

    items = [
        f"Rows: {df_before.shape[0]:,} → {df_after.shape[0]:,}  ({row_delta:+,})",
        f"Columns: {df_before.shape[1]} → {df_after.shape[1]}  ({col_delta:+,})",
        f"Missing values: {miss_before:,} → {miss_after:,}",
    ]

    label_map = {
        "missing":  "Missing value strategy",
        "outlier":  "Outlier strategy",
        "drop_col": "Dropped columns",
        "drop_dup": "Duplicate rows removed",
    }
    by_type: dict = {}
    for a in actions:
        by_type.setdefault(a["type"], []).append(f"{a['col']} → {a['detail']}")

    for t, entries in by_type.items():
        items.append(f"{label_map.get(t, t)} ({len(entries)}):")
        items.extend(f"  • {e}" for e in entries)

    if not actions:
        items.append("(ไม่มีการเปลี่ยนแปลง — ใช้ข้อมูล original)")

    append({
        "step":  "Data Cleaning",
        "icon":  "",
        "type":  "success",
        "items": items,
    })
    st.session_state[_ACTIONS_KEY] = []


# ── Transformation ────────────────────────────────────────────────────────────

def log_transformation(summary: dict, enc_decisions: dict, scaling_method: str, drop_cols: list):
    """
    summary: dict จาก apply_all (original_cols, dropped_cols, final_cols, scaling_method)
    enc_decisions: dict {col: method}  ← output จาก _render_encoding()
    """
    items = [
        f"Original columns: {summary.get('original_cols', '?')}",
        f"Dropped columns: {summary.get('dropped_cols', 0)}",
        f"After encoding: {summary.get('final_cols', '?')} columns",
        f"Scaling method: {scaling_method}",
    ]
    if drop_cols:
        items.append(f"Feature dropped ({len(drop_cols)}): {', '.join(drop_cols)}")

    enc_by_method: dict = {}
    for col, method in (enc_decisions.items() if isinstance(enc_decisions, dict) else []):
        enc_by_method.setdefault(method, []).append(col)
    for method, cols in enc_by_method.items():
        items.append(f"Encoding — {method}: {', '.join(cols)}")

    append({
        "step":  "Data Transformation",
        "icon":  "",
        "type":  "success",
        "items": items,
    })


# ── Model Process ─────────────────────────────────────────────────────────────

def log_model_process(result: dict, metrics: dict):
    """result: output จาก run_competition, metrics: output จาก get_metrics"""
    task_type  = result["task_type"]
    best_label = result["best_label"]
    competition = result["competition"]

    # leaderboard sorted
    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )

    items = [
        f"Task type: {task_type.upper()}",
        f"Models trained: {len(competition)}",
        f"Best model: {best_label}",
    ]

    if ranked:
        items.append("Leaderboard (CV Score):")
        for i, (_, v) in enumerate(ranked[:5]):
            medal = ["1.", "2.", "3.", "4.", "5."][i]
            items.append(f"  {medal} {v['label']}: {v['cv_score']:.4f} ±{v['cv_std']:.4f}")

    items.append("Test set metrics:")
    for name, val in metrics.items():
        items.append(f"  • {name}: {val}")

    append({
        "step":  "ML Process",
        "icon":  "",
        "type":  "success",
        "items": items,
    })
