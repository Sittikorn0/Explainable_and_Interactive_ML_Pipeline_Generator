import copy
from datetime import datetime
import streamlit as st

from backend.core.insight.trace_log import *
from backend.core.session.state import *

STEP_ORDER = ["upload", "cleaning", "eda", "transformation", "model_process", "insight"]
STEP_LABELS = {
    "upload": "Upload",
    "cleaning": "Cleaning",
    "eda": "Eda",
    "transformation": "Transform",
    "model_process": "Model Process",
    "insight": "Insight",
}

# Steps ที่ต้อง commit (EDA เป็น read-only ไม่ต้อง commit)
COMMIT_STEPS = ["upload", "cleaning", "transformation", "model_process"]

PIPELINE_KEY = "pipeline"

# session_state keys ที่ต้องลบเมื่อ rollback แต่ละ step
DOWNSTREAM_KEYS = {
    "cleaning": [
        "working_df", "working_df_source_shape", "cleaning_confirmed",
        "cleaning_summary_snapshot", "original_outlier_bounds",
        "treated_outlier_cols", "cleaning_actions", "original_dup_count",
        "dist_key", "dist_result", "dist_cache",
        "eda_dist_key", "eda_dist_result",
    ],
    "transformation": [
        "transformed_df", "trans_confirmed", "trans_summary",
        "trans_target_saved", "main_df_backup",
    ],
    "model_process": [
        "ml_result", "ml_metrics", "fi_data", "ml_task_type",
        "ml_scaling_used", "ml_leakage_warnings",
    ],
}


# คืน pipeline dict จาก session_state (init ถ้ายังไม่มี พร้อม auto-reconstruct snapshots) ใช้ทุก function ในไฟล์นี้
def get_pipeline() -> dict:
    if PIPELINE_KEY not in st.session_state:
        snapshots = {}
        # Auto-reconstruct snapshots for step indicator if session was reloaded
        if st.session_state.get("ml_result") is not None:
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
            snapshots["transformation"] = {"summary": {}, "timestamp": ""}
            snapshots["model_process"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("transformed_df") is not None or st.session_state.get("trans_confirmed"):
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
            snapshots["transformation"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("working_df") is not None or st.session_state.get("cleaning_confirmed"):
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("main_df") is not None:
            snapshots["upload"] = {"summary": {}, "timestamp": ""}

        st.session_state[PIPELINE_KEY] = {
            "snapshots": snapshots,
            "prev_snapshots": None,
            "rollback_from": None,
        }
    return st.session_state[PIPELINE_KEY]


# ลบ snapshot และ session_state keys ของ step ถัดๆ จาก step_idx ใช้ใน commit_step และ rollback_to
def clear_downstream(step_idx: int, include_self: bool = False):
    pipeline = get_pipeline()
    
    for s in COMMIT_STEPS:
        idx = STEP_ORDER.index(s)
        if (include_self and idx >= step_idx) or (not include_self and idx > step_idx):
            pipeline["snapshots"].pop(s, None)

            if s in DOWNSTREAM_KEYS:
                for key in DOWNSTREAM_KEYS[s]:
                    st.session_state.pop(key, None)

            step_names_map = {
                "upload": "Upload",
                "cleaning": "Data Cleaning",
                "transformation": "Data Transformation",
                "model_process": "Model Process",
            }
            if s in step_names_map:
                remove_steps_from([step_names_map[s]])
            
            if s == "model_process":
                delete_ml_cache()

# บันทึก snapshot ของ step ปัจจุบัน และเคลียร์ downstream ใช้ใน upload/cleaning/transformation/model_process page
def commit_step(step: str, summary: dict):
    pipeline = get_pipeline()
    step_idx = STEP_ORDER.index(step)
    
    has_downstream_done = any(STEP_ORDER.index(s) > step_idx and s in pipeline["snapshots"] for s in COMMIT_STEPS)
    if has_downstream_done:
        pipeline["prev_snapshots"] = copy.deepcopy(pipeline["snapshots"])
        pipeline["rollback_from"] = step

    pipeline["snapshots"][step] = {
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    
    # ล้างด่านถัดๆ ไป (Downstream)
    clear_downstream(step_idx, include_self=False)


# rollback pipeline กลับไปที่ step ที่ระบุ เคลียร์ downstream และ restore df ใช้ใน diff view / rollback button
def rollback_to(step: str):
    pipeline = get_pipeline()
    step_idx = STEP_ORDER.index(step)

    pipeline["prev_snapshots"] = copy.deepcopy(pipeline["snapshots"])
    pipeline["rollback_from"] = step

    clear_downstream(step_idx, include_self=True)

    if step == "transformation" and "_main_df_backup" in st.session_state:
        st.session_state["main_df"] = st.session_state["_main_df_backup"]

    if step in ["upload", "cleaning"]:
        df, _ = load_from_local()
        if df is not None:
            st.session_state["main_df"] = df


# คืน status แต่ละ step (done/current/locked) ใช้ใน navigation bar / step indicator
def get_step_status() -> dict[str, str]:
    pipeline = get_pipeline()
    snapshots = pipeline["snapshots"]
    has_data = st.session_state.get("main_df") is not None

    if not has_data:
        return {s: ("current" if s == "upload" else "locked") for s in STEP_ORDER}

    status = {}
    for s in STEP_ORDER:
        if s == "upload":
            status[s] = "done" if has_data else "current"
        elif s == "eda":
            if any(step in snapshots for step in ["transformation", "model_process"]):
                status[s] = "done"
            else:
                status[s] = "current" if has_data else "locked"
        elif s == "insight":
            status[s] = "current" if "model_process" in snapshots else "locked"
        elif s in snapshots:
            status[s] = "done"
        else:
            prev_s = COMMIT_STEPS[COMMIT_STEPS.index(s) - 1] if s in COMMIT_STEPS and COMMIT_STEPS.index(s) > 0 else "upload"

            if prev_s == "upload":
                status[s] = "current" if has_data else "locked"
            elif prev_s in snapshots or status.get(prev_s) == "current":
                status[s] = "current"
            else:
                status[s] = "locked"

    if "current" not in status.values():
        for s in STEP_ORDER:
            if status[s] == "locked":
                status[s] = "current"
                break

    return status


# คืน comparison dict (prev vs curr snapshots) สำหรับแสดง diff view หลัง rollback
def get_comparison() -> dict | None:
    pipeline = get_pipeline()
    prev = pipeline.get("prev_snapshots")
    if not prev:
        return None

    curr = pipeline["snapshots"]
    comparison = {}
    for step in COMMIT_STEPS:
        p = prev.get(step, {}).get("summary")
        c = curr.get(step, {}).get("summary")
        if p or c:
            comparison[step] = {"prev": p, "curr": c}
    return comparison


# ล้าง prev_snapshots และ rollback_from หลัง user dismiss diff view
def clear_comparison():
    pipeline = get_pipeline()
    pipeline["prev_snapshots"] = None
    pipeline["rollback_from"] = None


# ตรวจว่ามี diff view ที่รอแสดงผลอยู่หรือไม่ ใช้ใน navigation / page render
def has_comparison() -> bool:
    pipeline = get_pipeline()
    return pipeline.get("prev_snapshots") is not None
