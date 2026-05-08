"""
explainable/features/pipeline_state.py
จัดการ Pipeline Snapshot, Rollback, Comparison
"""
import copy
from datetime import datetime
import streamlit as st

STEP_ORDER = ["upload", "cleaning", "eda", "transformation", "ml_process", "explainable"]
STEP_LABELS = {
    "upload":         "Upload",
    "cleaning":       "Cleaning",
    "eda":            "EDA",
    "transformation": "Transform",
    "ml_process":     "ML Process",
    "explainable":    "Explainable",
}

# Steps ที่ต้อง commit (EDA เป็น read-only ไม่ต้อง commit)
COMMIT_STEPS = ["upload", "cleaning", "transformation", "ml_process"]

_PIPELINE_KEY = "_pipeline"

# session_state keys ที่ต้องลบเมื่อ rollback แต่ละ step
_DOWNSTREAM_KEYS = {
    "cleaning": [
        "working_df", "working_df_source_shape", "cleaning_confirmed",
        "cleaning_summary_snapshot", "original_outlier_bounds",
        "_treated_outlier_cols", "_cleaning_actions", "original_dup_count",
        "_dist_key", "_dist_result", "_dist_cache",
        "_eda_dist_key", "_eda_dist_result",
    ],
    "transformation": [
        "transformed_df", "trans_confirmed", "trans_summary",
        "_trans_target_saved", "_main_df_backup",
    ],
    "ml_process": [
        "ml_result", "ml_metrics", "_fi_data", "ml_task_type",
        "_ml_scaling_used", "_ml_leakage_warnings",
    ],
}


def _get_pipeline() -> dict:
    if _PIPELINE_KEY not in st.session_state:
        snapshots = {}
        # Auto-reconstruct snapshots for step indicator if session was reloaded
        if st.session_state.get("ml_result") is not None:
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
            snapshots["transformation"] = {"summary": {}, "timestamp": ""}
            snapshots["ml_process"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("transformed_df") is not None or st.session_state.get("trans_confirmed"):
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
            snapshots["transformation"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("working_df") is not None or st.session_state.get("cleaning_confirmed"):
            snapshots["upload"] = {"summary": {}, "timestamp": ""}
            snapshots["cleaning"] = {"summary": {}, "timestamp": ""}
        elif st.session_state.get("main_df") is not None:
            snapshots["upload"] = {"summary": {}, "timestamp": ""}

        st.session_state[_PIPELINE_KEY] = {
            "snapshots": snapshots,
            "prev_snapshots": None,
            "rollback_from": None,
        }
    return st.session_state[_PIPELINE_KEY]


def _clear_downstream(step_idx: int, include_self: bool = False):
    """ล้างข้อมูล Snapshots และ Session State ของด่านถัดๆ ไป (Downstream)"""
    pipeline = _get_pipeline()
    
    for s in COMMIT_STEPS:
        idx = STEP_ORDER.index(s)
        if (include_self and idx >= step_idx) or (not include_self and idx > step_idx):
            # 1. ลบ Snapshot
            pipeline["snapshots"].pop(s, None)
            
            # 2. ลบ Session State Keys
            if s in _DOWNSTREAM_KEYS:
                for key in _DOWNSTREAM_KEYS[s]:
                    st.session_state.pop(key, None)
            
            # 3. ลบ Physical Cache (เช่น ML Results)
            if s == "ml_process":
                from data_prepare.loading_data import delete_ml_cache
                delete_ml_cache()

def commit_step(step: str, summary: dict):
    """บันทึก summary snapshot และล้างด่าน downstream (เนื่องจากข้อมูลเปลี่ยน)"""
    pipeline = _get_pipeline()
    step_idx = STEP_ORDER.index(step)
    
    pipeline["snapshots"][step] = {
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    
    # ล้างด่านถัดๆ ไป (Downstream)
    _clear_downstream(step_idx, include_self=False)


def rollback_to(step: str):
    """ย้อนกลับไปแก้ไข step ที่ระบุ — ลบ downstream state ทั้งหมด"""
    pipeline = _get_pipeline()
    step_idx = STEP_ORDER.index(step)

    # เก็บ snapshot ปัจจุบันไว้เปรียบเทียบ
    pipeline["prev_snapshots"] = copy.deepcopy(pipeline["snapshots"])
    pipeline["rollback_from"] = step

    # ล้างข้อมูลของด่านนี้และด่านถัดไปทั้งหมด
    _clear_downstream(step_idx, include_self=True)

    # Special: คืนค่า main_df จาก backup ถ้ากำลังถอยกลับไปที่จุดเริ่มของ transformation
    if step == "transformation" and "_main_df_backup" in st.session_state:
        st.session_state["main_df"] = st.session_state["_main_df_backup"]

    # ถ้าถอยไปถึง Upload/Cleaning ให้โหลด data ใหม่จาก local เพื่อความชัวร์ (ป้องกัน state ค้าง)
    if step in ["upload", "cleaning"]:
        from data_prepare.loading_data import load_from_local
        filename = st.session_state.get("last_uploaded_file")
        if filename:
            df = load_from_local(filename)
            if df is not None:
                st.session_state["main_df"] = df

    # ลบ trace_log entries ของ downstream
    from explainable.state_manager.trace_log import remove_steps_from
    step_names_map = {
        "cleaning": "Data Cleaning",
        "transformation": "Data Transformation",
        "ml_process": "ML Process",
    }
    steps_to_remove = []
    for s in COMMIT_STEPS:
        if STEP_ORDER.index(s) >= step_idx and s in step_names_map:
            steps_to_remove.append(step_names_map[s])
    if steps_to_remove:
        remove_steps_from(steps_to_remove)


def get_step_status() -> dict[str, str]:
    """คืน status ของแต่ละ step: 'done' | 'current' | 'locked'"""
    pipeline = _get_pipeline()
    snapshots = pipeline["snapshots"]
    has_data = st.session_state.get("main_df") is not None

    if not has_data:
        return {s: ("current" if s == "upload" else "locked") for s in STEP_ORDER}

    status = {}
    for s in STEP_ORDER:
        if s == "upload":
            status[s] = "done" if has_data else "current"
        elif s == "eda":
            # EDA จะถือว่า 'done' ถ้ามีการทำด่านที่อยู่ถัดจากมันไปแล้ว (เช่น Transformation หรือ ML Process)
            # แต่ถ้ายังไม่ถึง ให้เป็น 'current' เมื่อ Cleaning เสร็จ
            if any(step in snapshots for step in ["transformation", "ml_process"]):
                status[s] = "done"
            elif "cleaning" in snapshots:
                status[s] = "current"
            else:
                status[s] = "locked"
        elif s == "explainable":
            # สำหรับด่านสุดท้าย ให้เป็น 'current' พร้อมใช้งานเมื่อ ML Process เสร็จ
            status[s] = "current" if "ml_process" in snapshots else "locked"
        elif s in snapshots:
            status[s] = "done"
        else:
            # หา step แรกที่ยังไม่ได้ทำ
            prev_s = COMMIT_STEPS[COMMIT_STEPS.index(s) - 1] if s in COMMIT_STEPS and COMMIT_STEPS.index(s) > 0 else "upload"
            if prev_s == "upload":
                status[s] = "current" if has_data else "locked"
            elif prev_s in snapshots:
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
    pipeline = _get_pipeline()
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
    pipeline = _get_pipeline()
    pipeline["prev_snapshots"] = None
    pipeline["rollback_from"] = None


def has_comparison() -> bool:
    pipeline = _get_pipeline()
    return pipeline.get("prev_snapshots") is not None
