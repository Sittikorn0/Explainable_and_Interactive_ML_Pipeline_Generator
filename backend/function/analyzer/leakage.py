import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Function
# คำนวณ normalized mutual information ระหว่าง feature กับ target ใช้ภายใน analyze_leakage
def mutual_info_score(col_vals: np.ndarray, y_vals: np.ndarray,
                       is_classif: bool) -> float:
    try:
        if is_classif:
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(
                col_vals.reshape(-1, 1), y_vals, discrete_features=False, random_state=0
            )[0]
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(
                col_vals.reshape(-1, 1), y_vals, random_state=0
            )[0]
        # normalize: MI / H(y)  clamp 0..1
        y_series = pd.Series(y_vals)
        counts   = y_series.value_counts(normalize=True)
        h_y      = float(-(counts * np.log(counts + 1e-12)).sum())
        return float(np.clip(mi / (h_y + 1e-12), 0, 1))
    except Exception:
        return 0.0

# ตรวจหา column ที่อาจทำให้เกิด Data Leakage (Pearson/Spearman/MI/Bijective) คืน list[dict] ใช้ใน analyze_all
def analyze_leakage(df: pd.DataFrame, target_col: str) -> list[dict]:
    results: list[dict] = []
    y        = df[target_col]
    is_classif = not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 20
    seen: set[str] = set()

    for col in df.columns:
        if col == target_col:
            continue

        reasons:  list[str] = []
        scores:   list[float] = [] # เก็บค่าคะแนนต่างๆ เพื่อประเมิน severity
        severity  = "low"

        # 0. Identity Check
        if df[col].astype(str).equals(y.astype(str)):
            severity = "high"
            reasons.append("คอลัมน์นี้เป็น**ข้อมูลชุดเดียวกัน**กับ Target (Identity)")

        # Prepare Data for Calculation
        if pd.api.types.is_numeric_dtype(df[col]):
            x_num = df[col].fillna(df[col].median())
        else:
            x_num = pd.Series(LabelEncoder().fit_transform(df[col].astype(str)))

        if pd.api.types.is_numeric_dtype(y):
            y_num = y.fillna(y.median())
        else:
            y_num = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

        # 1. Pearson & Spearman Correlation
        try:
            if x_num.std() > 0 and y_num.std() > 0:
                p_corr = abs(float(x_num.corr(y_num, method='pearson')))
                s_corr = abs(float(x_num.corr(y_num, method='spearman')))
                max_corr = max(p_corr, s_corr)
                if max_corr >= 0.60:
                    scores.append(max_corr)
                    reasons.append(f"ความสัมพันธ์ (Correlation) สูง: **{max_corr:.4f}**")
        except Exception:
            pass

        # 2. Mutual Information
        try:
            mi = mutual_info_score(x_num.values.reshape(-1, 1), y_num.values, is_classif)
            if mi >= 0.60:
                scores.append(mi)
                reasons.append(f"อำนาจการทำนาย (Mutual Info): **{mi:.4f}**")
        except Exception:
            pass

        # 3. Bijective Mapping
        if df[col].nunique() == y.nunique() and df[col].nunique() >= 2:
            try:
                pairs = set(zip(df[col].astype(str), y.astype(str)))
                if len(pairs) == df[col].nunique():
                    severity = "high"
                    reasons.append("มี One-to-one mapping  ทายผลลัพธ์ได้แม่นยำ 100%")
            except Exception:
                pass

        # Evaluate Severity
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score >= 0.90 or any(s >= 0.95 for s in scores):
                severity = "high"
            elif avg_score >= 0.60:
                if severity != "high":
                    severity = "medium"

        # 4. Name Similarity
        col_c    = col.lower().replace("_", "").replace("-", "")
        target_c = target_col.lower().replace("_", "").replace("-", "")
        if (col_c in target_c or target_c in col_c) and col_c != target_col.lower():
            if severity == "low":
                severity = "medium"
            reasons.append(f"ชื่อคอลัมน์คล้ายเป้าหมาย ('{target_col}')")

        if reasons and col not in seen:
            seen.add(col)
            results.append({
                "col": col, "reasons": reasons,
                "severity": severity,
            })

    order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: order[x["severity"]])
    return results


# wrapper ของ analyze_leakage คืน list[str] formatted สำหรับแสดงใน UI ใช้ใน transformation_page
def detect_leakage(df: pd.DataFrame, target_col: str) -> list[str]:
    return [
        f"**{item['col']}**  " + ", ".join(item["reasons"])
        for item in analyze_leakage(df, target_col)
    ]