# Libraries
import pandas as pd
import streamlit as st
import json
import os

from backend.core.session.session_manager import trace_log_path


LOG_KEY     = "trace_log"
ACTIONS_KEY = "cleaning_actions"

def persist_log():
    """Helper to save the trace log and actions to disk."""
    path = trace_log_path()
    data = {
        "log": st.session_state.get(LOG_KEY, []),
        "actions": st.session_state.get(ACTIONS_KEY, [])
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error persisting trace log: {e}")

def restore_log():
    """Helper to restore the trace log from disk."""
    path = trace_log_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            st.session_state[LOG_KEY] = data.get("log", [])
            st.session_state[ACTIONS_KEY] = data.get("actions", [])
    except Exception as e:
        print(f"Error restoring trace log: {e}")

# ── Core ──────────────────────────────────────────────────────────────────────

def ensure_restored():
    """Ensures session state is synced with disk before any modification."""
    if not st.session_state.get(LOG_KEY) and os.path.exists(trace_log_path()):
        restore_log()

def append(entry: dict):
    ensure_restored()
    log = st.session_state.setdefault(LOG_KEY, [])
    # ค้นหาว่ามี step นี้อยู่แล้วหรือไม่ ถ้ามีให้แทนที่ (Update) ถ้าไม่มีให้เพิ่มใหม่ (Append)
    existing_idx = next((i for i, e in enumerate(log) if e.get("step") == entry.get("step")), None)
    if existing_idx is not None:
        log[existing_idx] = entry
    else:
        log.append(entry)
    persist_log()

def get_log() -> list[dict]:
    # If session is fresh but disk has log, restore it
    if not st.session_state.get(LOG_KEY) and os.path.exists(trace_log_path()):
        restore_log()
    return st.session_state.get(LOG_KEY, [])

def clear():
    """เรียกเมื่อ upload ไฟล์ใหม่"""
    st.session_state[LOG_KEY]     = []
    st.session_state[ACTIONS_KEY] = []
    persist_log()

def remove_steps_from(step_names: list[str]):
    """ลบ trace entries ของ step ที่ระบุ (ใช้ตอน rollback)"""
    ensure_restored()
    log = st.session_state.get(LOG_KEY, [])
    st.session_state[LOG_KEY] = [e for e in log if e.get("step") not in step_names]
    persist_log()


# ── Upload ────────────────────────────────────────────────────────────────────

def log_upload(df, file_name: str, target_col: str, task_hint: str,
               target_reasons: list = None):
    n_numeric = df.select_dtypes(include="number").shape[1]
    n_categ   = df.select_dtypes(include=["object", "category"]).shape[1]
    n_miss    = int(df.isnull().sum().sum())

    items = [
        f"ไฟล์: {file_name}",
        f"ขนาด: {df.shape[0]:,} แถว × {df.shape[1]} คอลัมน์",
        f"ตัวเลข: {n_numeric} คอลัมน์  |  ข้อความ: {n_categ} คอลัมน์",
        f"ข้อมูลที่ขาดหาย: {n_miss:,} ช่อง",
        f"เป้าหมายที่ต้องทำนาย: {target_col}  →  {task_hint}",
    ]

    explanations = []

    # ทำไมเลือก target นี้
    if target_reasons:
        explanations.append(f"ทำไมเลือก '{target_col}' ให้เป็นสิ่งที่ต้องทำนาย?")
        for r in target_reasons:
            explanations.append(f"  → {r}")

    # ทำไมเป็น classification / regression
    y = df[target_col]
    if task_hint == "classification":
        if not pd.api.types.is_numeric_dtype(y):
            explanations.append(
                f"ทำไมเป็นงาน Classification? เพราะ '{target_col}' เป็นข้อความ "
                f"(มี {y.nunique()} กลุ่ม) ระบบจึงต้อง \"แยกกลุ่ม\" ไม่ใช่ทำนายตัวเลข"
            )
        else:
            explanations.append(
                f"ทำไมเป็นงาน Classification? เพราะ '{target_col}' เป็นตัวเลข "
                f"แต่มีแค่ {y.nunique()} ค่า จึงถือว่าเป็นกลุ่มที่นับได้"
            )
    else:
        explanations.append(
            f"ทำไมเป็นงาน Regression? เพราะ '{target_col}' เป็นตัวเลขต่อเนื่อง "
            f"({y.nunique()} ค่า) ระบบจึงต้อง \"ทำนายค่า\" แทนการแยกกลุ่ม"
        )

    # สถานะข้อมูลขาดหาย
    miss_pct = n_miss / df.size * 100 if df.size > 0 else 0
    if miss_pct > 5:
        explanations.append(
            f"⚠ ข้อมูลขาดหายค่อนข้างเยอะ ({miss_pct:.1f}%) "
            "ควรจัดการก่อนสร้างโมเดล ไม่งั้นผลอาจคลาดเคลื่อน"
        )
    elif n_miss == 0:
        explanations.append("ไม่มีข้อมูลขาดหาย พร้อมใช้งานได้เลย")
    else:
        explanations.append(
            f"ข้อมูลขาดหายเล็กน้อย ({miss_pct:.1f}%) "
            "ระบบจะจัดการให้ในขั้นตอนทำความสะอาด"
        )

    append({"step": "Upload", "items": items, "explanations": explanations})


# ── Cleaning ──────────────────────────────────────────────────────────────────

def track_cleaning(action_type: str, col: str, detail: str, explanation: str = ""):
    ensure_restored()
    actions = st.session_state.setdefault(ACTIONS_KEY, [])
    actions = [a for a in actions if not (a["col"] == col and a["type"] == action_type)]
    actions.append({"type": action_type, "col": col, "detail": detail, "explanation": explanation})
    st.session_state[ACTIONS_KEY] = actions
    persist_log()

def track_cleaning_bulk(action_type: str, cols: list, detail: str, explanation: str = ""):
    for col in cols:
        track_cleaning(action_type, col, detail, explanation)
    persist_log()


def commit_cleaning(df_before, df_after):
    actions   = st.session_state.get(ACTIONS_KEY, [])
    row_delta = df_after.shape[0] - df_before.shape[0]
    col_delta = df_after.shape[1] - df_before.shape[1]
    miss_before = int(df_before.isnull().sum().sum())
    miss_after  = int(df_after.isnull().sum().sum())

    items = [
        f"จำนวนแถว: ก่อน {df_before.shape[0]:,} หลัง {df_after.shape[0]:,}  ({row_delta:+,})",
        f"จำนวนคอลัมน์: ก่อน {df_before.shape[1]} หลัง {df_after.shape[1]}  ({col_delta:+,})",
        f"ข้อมูลขาดหาย: ก่อน {miss_before:,} หลัง {miss_after:,}",
    ]

    label_map = {
        "missing":  "วิธีจัดการข้อมูลขาดหาย",
        "outlier":  "วิธีจัดการค่าผิดปกติ",
        "drop_col": "คอลัมน์ที่ลบออก",
        "drop_dup": "แถวซ้ำที่ลบออก",
    }
    by_type: dict = {}
    for a in actions:
        by_type.setdefault(a["type"], []).append(a)
    for t, entries in by_type.items():
        items.append(f"{label_map.get(t, t)} ({len(entries)}):")
        items.extend(f"  • {e['col']} → {e['detail']}" for e in entries)

    if not actions:
        items.append("(ไม่มีการเปลี่ยนแปลง ใช้ข้อมูลเดิม)")

    explanations = []

    for a in by_type.get("missing", []):
        if a.get("explanation"):
            explanations.append(f"[{a['col']}] {a['explanation']}")
        else:
            d = a["detail"].lower()
            if "median" in d:
                explanations.append(f"ทำไมเติมค่ากลาง (Median) ให้ '{a['col']}'? เพราะค่ากลางไม่ถูกดึงไปตามค่าที่สูงหรือต่ำผิดปกติ")
            elif "mean" in d:
                explanations.append(f"ทำไมเติมค่าเฉลี่ย (Mean) ให้ '{a['col']}'? เพราะข้อมูลกระจายตัวสม่ำเสมอ ค่าเฉลี่ยจึงเป็นตัวแทนที่ดี")
            elif "mode" in d or "frequent" in d:
                explanations.append(f"ทำไมเติมค่าที่พบบ่อยสุด (Mode) ให้ '{a['col']}'? เพราะเป็นข้อมูลข้อความ ใช้ค่าเฉลี่ยไม่ได้")
            elif "drop" in d:
                explanations.append(f"ทำไมลบแถวที่ข้อมูลขาดหายของ '{a['col']}'? เพราะมีจำนวนน้อย ลบออกแล้วไม่กระทบภาพรวม")

    for a in by_type.get("outlier", []):
        if a.get("explanation"):
            explanations.append(f"[{a['col']}] {a['explanation']}")
        else:
            d = a["detail"].lower()
            if "clip" in d:
                explanations.append(f"ทำไมตัดค่าผิดปกติของ '{a['col']}' ให้อยู่ในขอบเขต? เพราะค่าที่สูงหรือต่ำเกินไปจะทำให้โมเดลสับสน")
            elif "drop" in d or "remove" in d:
                explanations.append(f"ทำไมลบค่าผิดปกติของ '{a['col']}'? เพราะค่านั้นอาจเกิดจากความผิดพลาดในการบันทึก")

    if "drop_dup" in by_type:
        explanations.append("ทำไมลบแถวที่ซ้ำกัน? เพราะข้อมูลซ้ำทำให้โมเดลให้ความสำคัญกับข้อมูลนั้นมากเกินจริง อาจทำให้ผลไม่แม่นยำ")

    if by_type.get("drop_col"):
        for a in by_type["drop_col"]:
            if a.get("explanation"):
                explanations.append(f"[{a['col']}] {a['explanation']}")
            else:
                explanations.append(f"ทำไมลบคอลัมน์ '{a['col']}'? เพราะไม่มีข้อมูลที่เป็นประโยชน์ เก็บไว้จะเป็นสัญญาณรบกวนให้โมเดล")

    if not actions:
        explanations.append("ไม่ต้องทำความสะอาด เพราะข้อมูลอยู่ในสภาพดีอยู่แล้ว")
    else:
        explanations.append("หลักการ: ต้องทำความสะอาดข้อมูลก่อนสร้างโมเดลเสมอ เพราะข้อมูลที่สะอาดจะช่วยให้โมเดลเรียนรู้ได้แม่นยำขึ้น")

    append({"step": "Data Cleaning", "items": items, "explanations": explanations})
    st.session_state[ACTIONS_KEY] = []


# ── Transformation ────────────────────────────────────────────────────────────

SCALING_REASONS = {
    "standard_scaler": (
        "ปรับให้ข้อมูลมีค่าเฉลี่ยเป็น 0 และกระจายตัวเท่ากัน "
        "เหมาะกับข้อมูลที่กระจายแบบระฆังคว่ำ"
    ),
    "minmax_scaler": (
        "ย่อค่าทุกตัวให้อยู่ในช่วง 0 ถึง 1 "
        "เหมาะเมื่อข้อมูลไม่มีค่าผิดปกติรุนแรง"
    ),
    "robust_scaler": (
        "ใช้ค่ากลางและค่าพิสัยกลาง ทนทานต่อค่าผิดปกติ "
        "ดีกว่าวิธีมาตรฐานเมื่อข้อมูลมีค่าสุดโต่ง"
    ),
    "log_transform": (
        "แปลงข้อมูลด้วย Log เพื่อลดความเบ้ของข้อมูลที่กระจุกตัวด้านใดด้านหนึ่ง"
    ),
    "no_scaling": (
        "ไม่ปรับขนาด เพราะโมเดลแบบต้นไม้ (เช่น Random Forest, XGBoost) "
        "ตัดสินใจจากการเปรียบเทียบค่า ไม่ได้สนใจขนาดของตัวเลข"
    ),
}

ENCODING_REASONS = {
    "onehot": "มีค่าไม่กี่แบบ จึงสร้างคอลัมน์ใหม่แทนแต่ละค่า เพื่อไม่ให้โมเดลเข้าใจว่ามีลำดับ",
    "label":  "มีค่าหลายแบบเกินไป จึงแปลงเป็นตัวเลข 0, 1, 2, ... แทน",
    "ordinal": "ข้อมูลมีลำดับชัดเจน (เช่น ต่ำ < กลาง < สูง) จึงแปลงเป็นตัวเลขตามลำดับ",
    "drop":   "ไม่เหมาะจะนำมาสร้างโมเดล เช่น รหัสหรือข้อความอิสระ",
    "skip":   "เป็นตัวเลขอยู่แล้ว ไม่ต้องแปลง",
}


def log_transformation(summary: dict, enc_decisions: dict, scaling_method: str, drop_cols: list,
                       scaling_reason: str = "", enc_reasons: dict = None):
    """
    enc_reasons: {col: reason_string} จาก rule engine (optional)
    scaling_reason: explanation string จาก rule engine (optional)
    """
    items = [
        f"คอลัมน์เริ่มต้น: {summary.get('original_cols', '?')}",
        f"คอลัมน์ที่ลบ: {summary.get('dropped_cols', 0)}",
        f"หลังแปลงข้อมูล: {summary.get('final_cols', '?')} คอลัมน์",
        f"วิธีปรับขนาด: {scaling_method}",
    ]
    if drop_cols:
        items.append(f"คอลัมน์ที่ตัดออก ({len(drop_cols)}): {', '.join(drop_cols)}")

    enc_by_method: dict = {}
    for col, method in (enc_decisions.items() if isinstance(enc_decisions, dict) else []):
        enc_by_method.setdefault(method, []).append(col)
    for method, cols in enc_by_method.items():
        items.append(f"การแปลงข้อความ  {method}: {', '.join(cols)}")

    explanations = []

    # Scaling reason — ใช้จาก rule engine ถ้ามี ไม่งั้น fallback hardcode
    scl_reason = scaling_reason or SCALING_REASONS.get(scaling_method, f"ใช้ {scaling_method}")
    explanations.append(f"ทำไมปรับขนาดด้วย {scaling_method}? เหตุผลคือ {scl_reason}")

    if scaling_method != "no_scaling":
        explanations.append(
            "ทำไมต้องปรับขนาดข้อมูล? เพราะถ้าคอลัมน์หนึ่งมีค่าเป็นหลักล้าน "
            "แต่อีกคอลัมน์มีค่าแค่ 0-100 โมเดลจะให้ความสำคัญกับตัวเลขใหญ่มากเกินไป "
            "ปรับขนาดช่วยให้ทุกคอลัมน์มีน้ำหนักเท่าเทียมกัน"
        )

    # Encoding reasons — ใช้จาก rule engine รายคอลัมน์ถ้ามี
    enc_reasons = enc_reasons or {}
    for method, cols in enc_by_method.items():
        # ดึง reason จาก rule engine (ใช้ reason ของ col แรกในกลุ่มเป็นตัวแทน)
        rule_reason = next((enc_reasons[c] for c in cols if c in enc_reasons), "")
        reason = rule_reason or ENCODING_REASONS.get(method, f"ใช้ {method}")
        cols_str = ", ".join(cols[:3])
        if len(cols) > 3:
            cols_str += f" อีก {len(cols)-3} ตัว"
        explanations.append(f"ทำไมแปลง {cols_str} ด้วย {method}? เหตุผลคือ {reason}")

    if enc_by_method:
        explanations.append(
            "ทำไมต้องแปลงข้อความเป็นตัวเลข? เพราะโมเดลคำนวณได้เฉพาะตัวเลขเท่านั้น "
            "ข้อความเช่น \"แดง\" หรือ \"น้ำเงิน\" ต้องถูกแปลงก่อน"
        )

    if drop_cols:
        explanations.append(
            f"ทำไมตัดออก {len(drop_cols)} คอลัมน์? "
            "เพราะเป็นข้อมูลที่ไม่ช่วยในการทำนาย เช่น รหัสที่ไม่ซ้ำกัน "
            "หรือข้อมูลที่ขาดหายเยอะเกินไป เก็บไว้จะรบกวนการเรียนรู้ของโมเดล"
        )

    append({"step": "Data Transformation", "items": items, "explanations": explanations})


# ── Model Process ─────────────────────────────────────────────────────────────

def log_model_process(result: dict, metrics: dict):
    task_type   = result["task_type"]
    best_label  = result["best_label"]
    competition = result["competition"]

    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )

    items = [
        f"ประเภทงาน: {task_type.upper()}",
        f"จำนวนโมเดลที่ทดสอบ: {len(competition)}",
        f"โมเดลที่ดีที่สุด: {best_label}",
    ]

    if ranked:
        items.append("อันดับ (คะแนน CV):")
        for i, (_, v) in enumerate(ranked[:5]):
            items.append(f"  {i+1}. {v['label']}: {v['cv_score']:.4f} ±{v['cv_std']:.4f}")

    items.append("ผลทดสอบจริง:")
    for name, val in metrics.items():
        items.append(f"  • {name}: {val}")

    explanations = []

    # ทำไมเลือกโมเดลนี้
    if len(ranked) >= 2:
        best_score = ranked[0][1]["cv_score"]
        second = ranked[1][1]
        diff = best_score - second["cv_score"]
        explanations.append(
            f"ทำไมเลือก {best_label}? เพราะได้คะแนนสูงที่สุด ({best_score:.4f}) "
            f"ชนะอันดับสอง ({second['label']}) อยู่ {diff:.4f} คะแนน"
        )
        if diff < 0.01:
            explanations.append(
                "⚠ คะแนนห่างกันน้อยมากอาจเลือกโมเดลที่อธิบายได้ง่ายกว่าแทน "
            )

    # ทำไมใช้ Cross-Validation
    explanations.append(
        "ทำไมใช้ Cross-Validation ตัดสิน? การทำงานคือระบบแบ่งข้อมูลเป็น 5 ชุด "
        "วนสลับกันทดสอบทุกชุด แล้วเอาคะแนนมาเฉลี่ย "
        "ให้ผลที่เชื่อถือได้มากกว่าทดสอบแค่ครั้งเดียว"
    )

    # ทำไมแบ่ง 80/20
    explanations.append(
        "ทำไมแบ่งข้อมูล 80/20? ใช้ 80% สอนโมเดล เก็บ 20% ไว้ทดสอบ "
        "เหมือนให้นักเรียนทำข้อสอบที่ไม่เคยเห็น "
        "ถ้าทำได้ดี แสดงว่าเข้าใจจริง ไม่ใช่แค่ท่องจำ"
    )

    # แปลผลคะแนน
    if task_type == "classification":
        acc = metrics.get("Accuracy")
        f1  = metrics.get("F1(Mac)")
        if isinstance(acc, (int, float)) and isinstance(f1, (int, float)):
            if acc > 0.9:
                explanations.append(f"ผลทดสอบ: Accuracy = {acc:.4f} อยู่ในเกณฑ์ดีมาก ทำนายถูกมากกว่า 90%")
            elif acc > 0.7:
                explanations.append(f"ผลทดสอบ: Accuracy = {acc:.4f} อยู่ในเกณฑ์ดี ยังพัฒนาต่อได้")
            else:
                explanations.append(
                    f"ผลทดสอบ: Accuracy = {acc:.4f} อยู่ในเกณฑ์ไม่ค่อยดี "
                    "อาจต้องปรับปรุงข้อมูลหรือเพิ่มข้อมูลเพิ่มเติม"
                )
            if abs(acc - f1) > 0.1:
                explanations.append(
                    f"⚠ Accuracy ({acc:.4f}) กับ F1 Macro ({f1:.4f}) ต่างกันมาก "
                    "แสดงว่าบางกลุ่มมีข้อมูลน้อยกว่ากลุ่มอื่นมาก "
                    "ควรดู F1 Macro เป็นหลัก เพราะให้น้ำหนักทุกกลุ่มเท่ากัน"
                )
    else:
        r2 = metrics.get("R² Score")
        if isinstance(r2, (int, float)):
            if r2 > 0.9:
                explanations.append(f"ผลทดสอบ: R² = {r2:.4f} อยู่ในเกณฑ์ดีมาก โมเดลสามารถอธิบายข้อมูลได้มากกว่า 90%")
            elif r2 > 0.5:
                explanations.append(f"ผลทดสอบ: R² = {r2:.4f} อยู่ในเกณฑ์พอใช้ได้ ยังปรับปรุงต่อได้")
            else:
                explanations.append(f"ผลทดสอบ: R² = {r2:.4f} อยู่ในเกณฑ์ยังอธิบายข้อมูลได้ไม่ดีนัก อาจต้องเพิ่มข้อมูล")

    failed = [(k, v) for k, v in competition.items() if v["cv_score"] is None]
    if failed:
        names = ", ".join(v["label"] for _, v in failed)
        explanations.append(
            f"ℹ มี {len(failed)} โมเดลที่ฝึกไม่สำเร็จ ({names}) "
            "อาจไม่เข้ากับข้อมูลชุดนี้ ระบบข้ามไปอัตโนมัติ"
        )

    append({"step": "Model Process", "items": items, "explanations": explanations})
