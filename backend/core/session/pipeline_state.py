import copy
from datetime import datetime
import streamlit as st

from backend.core.insight.trace_log import *
from backend.core.session.state import *
from backend.core.session.state import load_raw_data

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


def clear_downstream(step_idx: int, include_self: bool = False):
    """ล้างข้อมูล Snapshots และ Session State ของด่านถัดๆ ไป (Downstream)"""
    pipeline = get_pipeline()
    
    for s in COMMIT_STEPS:
        idx = STEP_ORDER.index(s)
        if (include_self and idx >= step_idx) or (not include_self and idx > step_idx):
            # 1. ลบ Snapshot
            pipeline["snapshots"].pop(s, None)
            
            # 2. ลบ Session State Keys
            if s in DOWNSTREAM_KEYS:
                for key in DOWNSTREAM_KEYS[s]:
                    st.session_state.pop(key, None)
            
            # 3. ลบ Trace Log ของด่านนั้นๆ
            step_names_map = {
                "upload": "Upload",
                "cleaning": "Data Cleaning",
                "transformation": "Data Transformation",
                "model_process": "Model Process",
            }
            if s in step_names_map:
                remove_steps_from([step_names_map[s]])
            
            # 3. ลบ Physical Cache (เช่น ML Results)
            if s == "model_process":
                delete_ml_cache()

def commit_step(step: str, summary: dict):
    """บันทึก summary snapshot และล้างด่าน downstream (เนื่องจากข้อมูลเปลี่ยน)"""
    pipeline = get_pipeline()
    step_idx = STEP_ORDER.index(step)
    
    # ก่อนจะล้างด่านถัดไป ให้เก็บของเดิมไว้ใน prev_snapshots ก่อน (เพื่อรองรับ Diff View)
    # เฉพาะกรณีที่มีด่านถัดไปทำเสร็จอยู่แล้วเท่านั้น
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


def rollback_to(step: str):
    """ย้อนกลับไปแก้ไข step ที่ระบุ  ลบ downstream state ทั้งหมด"""
    pipeline = get_pipeline()
    step_idx = STEP_ORDER.index(step)

    # เก็บ snapshot ปัจจุบันไว้เปรียบเทียบ
    pipeline["prev_snapshots"] = copy.deepcopy(pipeline["snapshots"])
    pipeline["rollback_from"] = step

    # ล้างข้อมูลของด่านนี้และด่านถัดไปทั้งหมด
    clear_downstream(step_idx, include_self=True)

    # Special: คืนค่า main_df จาก backup ถ้ากำลังถอยกลับไปที่จุดเริ่มของ transformation
    if step == "transformation" and "_main_df_backup" in st.session_state:
        st.session_state["main_df"] = st.session_state["_main_df_backup"]

    # ถ้าถอยไปถึง Cleaning ให้โหลด raw data (ก่อน cleaning) เพื่อให้ reset ได้จากต้นฉบับ
    if step == "cleaning":
        raw_df = load_raw_data()
        if raw_df is not None:
            st.session_state["main_df"] = raw_df
        else:
            df, _ = load_from_local()
            if df is not None:
                st.session_state["main_df"] = df
    elif step == "upload":
        df, _ = load_from_local()
        if df is not None:
            st.session_state["main_df"] = df


def get_step_status() -> dict[str, str]:
    """คืน status ของแต่ละ step: 'done' | 'current' | 'locked'"""
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
            # EDA จะถือว่า 'done' ถ้ามีการทำด่านที่อยู่ถัดจากมันไปแล้ว
            if any(step in snapshots for step in ["transformation", "model_process"]):
                status[s] = "done"
            else:
                # EDA พร้อมเสมอถ้ามีข้อมูล (ไม่ต้องรอ Cleaning)
                status[s] = "current" if has_data else "locked"
        elif s == "insight":
            # สำหรับด่านสุดท้าย ให้เป็น 'current' พร้อมใช้งานเมื่อ ML Process เสร็จ
            status[s] = "current" if "model_process" in snapshots else "locked"
        elif s in snapshots:
            status[s] = "done"
        else:
            # หา step ก่อนหน้าในลำดับความสำคัญ (Dependency)
            prev_s = COMMIT_STEPS[COMMIT_STEPS.index(s) - 1] if s in COMMIT_STEPS and COMMIT_STEPS.index(s) > 0 else "upload"
            
            if prev_s == "upload":
                status[s] = "current" if has_data else "locked"
            elif prev_s in snapshots or status.get(prev_s) == "current":
                # ถ้าด่านก่อนหน้าทำเสร็จแล้ว หรือด่านก่อนหน้าพร้อมทำ ด่านนี้ก็ควรจะ 'พร้อมทำ (current)' เช่นกัน
                status[s] = "current"
            else:
                status[s] = "locked"

    # ถ้าไม่มี current ให้หา step แรกที่ยังไม่ done
    if "current" not in status.values():
        for s in STEP_ORDER:
            if status[s] == "locked":
                status[s] = "current"
                break

    return status


def get_comparison() -> dict | None:
    """คืน dict ของ step → {prev, curr} สำหรับแสดง comparison"""
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


def clear_comparison():
    """ล้าง prev_snapshots เมื่อผู้ใช้ไม่ต้องการดู comparison แล้ว"""
    pipeline = get_pipeline()
    pipeline["prev_snapshots"] = None
    pipeline["rollback_from"] = None


def has_comparison() -> bool:
    pipeline = get_pipeline()
    return pipeline.get("prev_snapshots") is not None
    pipeline["rollback_from"] = None


def has_comparison() -> bool:
    pipeline = get_pipeline()
    return pipeline.get("prev_snapshots") is not None
