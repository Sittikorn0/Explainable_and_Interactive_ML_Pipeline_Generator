import numpy as np
import pandas as pd
from scipy.stats import skew


# ── Encoding Analysis ─────────────────────────────────────────

def analyze_encoding(df: pd.DataFrame, target_col: str) -> list[dict]:
    """
    วิเคราะห์ categorical columns แล้วแนะนำ encoding method

    Returns: list of {
        col, dtype, cardinality, recommended, options, reason, warning
    }
    """
    results = []
    cat_cols = [
        c for c in df.columns
        if c != target_col and df[c].dtype == object
    ]

    for col in cat_cols:
        cardinality = df[col].nunique()
        n_rows      = len(df)
        ratio       = cardinality / n_rows

        # ── เลือก recommended + เหตุผล ──────────────────────
        if cardinality == 2:
            recommended = "label_encoding"
            reason = (
                f"มีเพียง **2 categories** ({', '.join(df[col].dropna().unique().astype(str))}) "
                f"— Label Encoding (0/1) เพียงพอและไม่สิ้นเปลืองคอลัมน์"
            )
            warning = None

        elif cardinality <= 10:
            recommended = "one_hot_encoding"
            reason = (
                f"มี **{cardinality} categories** ซึ่งถือว่าน้อย (low cardinality) "
                f"— One-hot Encoding เหมาะที่สุดเพราะไม่สร้าง ordinal relationship "
                f"ที่ไม่มีอยู่จริง เช่น 'Bangkok=1, Chiang Mai=2' ซึ่ง model อาจเข้าใจผิดว่า "
                f"Chiang Mai 'มากกว่า' Bangkok"
            )
            warning = None

        elif cardinality <= 20:
            recommended = "one_hot_encoding"
            reason = (
                f"มี **{cardinality} categories** — One-hot ยังใช้ได้ "
                f"แต่จะเพิ่ม {cardinality-1} คอลัมน์ใหม่"
            )
            warning = f"จะสร้าง {cardinality-1} คอลัมน์ใหม่ อาจทำให้ dataset กว้างขึ้น"

        elif ratio > 0.5:
            recommended = "drop_column"
            reason = (
                f"มี **{cardinality} unique values** จากทั้งหมด {n_rows:,} rows "
                f"({ratio*100:.0f}%) — แทบทุก row มีค่าต่างกัน เช่น ID หรือชื่อ "
                f"คอลัมน์นี้ไม่มีประโยชน์สำหรับ ML เพราะ model ไม่สามารถเรียนรู้ pattern ได้"
            )
            warning = "คอลัมน์นี้อาจเป็น ID หรือ free-text ที่ไม่ควรใช้เป็น feature"

        else:
            recommended = "label_encoding"
            reason = (
                f"มี **{cardinality} categories** (high cardinality) "
                f"— Label Encoding ดีกว่า One-hot เพราะ One-hot จะสร้าง {cardinality-1} คอลัมน์ "
                f"ทำให้ dataset ใหญ่มากและ model ช้า (Curse of Dimensionality)"
            )
            warning = f"High cardinality: {cardinality} unique values"

        results.append({
            "col":         col,
            "cardinality": cardinality,
            "recommended": recommended,
            "options":     ["one_hot_encoding", "label_encoding", "ordinal_encoding", "drop_column"],
            "reason":      reason,
            "warning":     warning,
            "sample_values": list(df[col].dropna().unique()[:5]),
        })

    return results


# ── Scaling Analysis ──────────────────────────────────────────

def analyze_scaling(df: pd.DataFrame, target_col: str) -> dict:
    """
    วิเคราะห์ numeric columns แล้วแนะนำ scaling method

    Returns: {
        recommended, options, reason, column_stats, has_outliers, is_skewed
    }
    """
    num_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not num_cols:
        return {
            "recommended": "no_scaling",
            "options":     ["no_scaling"],
            "reason":      "ไม่มีคอลัมน์ตัวเลขที่ต้องทำ scaling",
            "column_stats": [],
            "has_outliers": False,
            "is_skewed":    False,
        }

    # วิเคราะห์แต่ละคอลัมน์
    col_stats       = []
    outlier_cols    = []
    skewed_cols     = []
    heavy_skew_cols = []

    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        col_skew = 0.0 if series.std() < 1e-10 else float(skew(series))

        # ตรวจ outlier ด้วย IQR
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr    = q3 - q1
        n_out  = int(((series < q1 - 1.5*iqr) | (series > q3 + 1.5*iqr)).sum())
        out_pct = n_out / len(series) * 100

        col_stats.append({
            "col":     col,
            "min":     round(float(series.min()), 3),
            "max":     round(float(series.max()), 3),
            "mean":    round(float(series.mean()), 3),
            "std":     round(float(series.std()), 3),
            "skew":    round(col_skew, 3),
            "outlier_pct": round(out_pct, 1),
        })

        if out_pct > 5:
            outlier_cols.append(col)
        if abs(col_skew) > 1:
            skewed_cols.append(col)
        if abs(col_skew) > 2:
            heavy_skew_cols.append(col)

    has_outliers    = len(outlier_cols) > 0
    is_skewed       = len(skewed_cols) > 0
    has_heavy_skew  = len(heavy_skew_cols) > 0

    # ── เลือก recommended ─────────────────────────────────────
    if has_outliers:
        recommended = "robust_scaler"
        reason = (
            f"พบ outlier ใน **{len(outlier_cols)} คอลัมน์** "
            f"({', '.join(outlier_cols[:3])}{'...' if len(outlier_cols)>3 else ''}) "
            f"— **Robust Scaler** เหมาะที่สุดเพราะใช้ **Median และ IQR** แทน Mean/Std "
            f"ทำให้ค่า extreme ไม่ดึง scale ให้เบี้ยว\n\n"
            f"เปรียบเทียบ: ถ้าใช้ Standard Scaler กับข้อมูลที่มี outlier "
            f"ค่าปกติส่วนใหญ่จะถูกบีบให้อยู่ในช่วงแคบมาก ทำให้ model แยกแยะได้ยาก"
        )
    elif has_heavy_skew:
        recommended = "log_transform"
        reason = (
            f"พบข้อมูล skewed รุนแรง (|skew| > 2) ใน **{len(heavy_skew_cols)} คอลัมน์** "
            f"({', '.join(heavy_skew_cols[:3])}{'...' if len(heavy_skew_cols)>3 else ''}) "
            f"— **Log Transform** ลด skewness ของข้อมูลก่อน แล้วตาม Standard Scaler\n\n"
            f"เหมาะกับข้อมูลอย่าง รายได้ ราคา จำนวน transaction ที่มี distribution แบบ long-tail"
        )
    elif is_skewed:
        recommended = "minmax_scaler"
        reason = (
            f"ข้อมูลมีการกระจายแบบ skewed ใน **{len(skewed_cols)} คอลัมน์** "
            f"({', '.join(skewed_cols[:3])}{'...' if len(skewed_cols)>3 else ''}) "
            f"— **MinMax Scaler** แปลงค่าให้อยู่ในช่วง [0, 1] "
            f"เหมาะกับข้อมูลที่ไม่ได้กระจายแบบ normal distribution\n\n"
            f"Standard Scaler สมมติว่าข้อมูลกระจายแบบ normal จึงไม่เหมาะในกรณีนี้"
        )
    else:
        recommended = "standard_scaler"
        reason = (
            f"ข้อมูล numeric ทั้งหมดมีการกระจายใกล้เคียง normal distribution "
            f"และไม่มี outlier มีนัย "
            f"— **Standard Scaler** (Z-score normalization) เหมาะที่สุด "
            f"แปลงให้ mean=0, std=1 ทำให้ทุก feature อยู่ในสเกลเดียวกัน "
            f"และ model ส่วนใหญ่ทำงานได้ดีที่สุดกับข้อมูลแบบนี้"
        )

    return {
        "recommended":    recommended,
        "options":        ["log_transform", "standard_scaler", "minmax_scaler", "robust_scaler", "no_scaling"],
        "reason":         reason,
        "column_stats":   col_stats,
        "has_outliers":   has_outliers,
        "is_skewed":      is_skewed,
        "has_heavy_skew": has_heavy_skew,
        "outlier_cols":   outlier_cols,
        "skewed_cols":    skewed_cols,
        "heavy_skew_cols": heavy_skew_cols,
    }


# ── Feature Selection Analysis ────────────────────────────────

def analyze_feature_selection(df: pd.DataFrame, target_col: str) -> dict:
    """
    วิเคราะห์และแนะนำ features ที่ควรตัดออก

    Returns: {
        drop_high_corr: [(col_a, col_b, corr)],
        drop_low_var:   [col],
        reason_corr, reason_var
    }
    """
    num_cols = [
        c for c in df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df[c])
    ]

    drop_high_corr = []
    drop_low_var   = []

    # ── High Correlation ──────────────────────────────────────
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr().abs()
        seen = set()
        for i, col_a in enumerate(num_cols):
            for col_b in num_cols[i+1:]:
                pair = tuple(sorted([col_a, col_b]))
                if pair in seen:
                    continue
                seen.add(pair)
                c = corr_matrix.loc[col_a, col_b]
                if c >= 0.85:
                    drop_high_corr.append({
                        "col_a": col_a,
                        "col_b": col_b,
                        "corr":  round(float(c), 3),
                        "drop":  col_b,   # แนะนำตัดตัวหลัง
                    })

    # ── Low Variance ─────────────────────────────────────────
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        cv = series.std() / (series.mean() + 1e-9)   # coefficient of variation
        if abs(cv) < 0.01:
            drop_low_var.append({
                "col": col,
                "std": round(float(series.std()), 6),
                "cv":  round(float(cv), 6),
            })

    reason_corr = (
        "คอลัมน์ที่มี **correlation ≥ 0.85** กับคอลัมน์อื่น "
        "มีข้อมูลซ้ำซ้อน (multicollinearity) — การเก็บทั้งคู่ไว้ทำให้ model "
        "ให้ความสำคัญกับข้อมูลนั้นมากเกินจริง และทำให้ interpret ผลได้ยาก"
    )
    reason_var = (
        "คอลัมน์ที่มี **variance ต่ำมาก** แทบไม่มีข้อมูลที่เป็นประโยชน์ "
        "เพราะค่าเกือบทั้งหมดเหมือนกัน model ไม่สามารถเรียนรู้ pattern จากคอลัมน์นี้ได้"
    )

    return {
        "drop_high_corr": drop_high_corr,
        "drop_low_var":   drop_low_var,
        "reason_corr":    reason_corr,
        "reason_var":     reason_var,
    }


# ── Master Analyze ────────────────────────────────────────────

def analyze_all(df: pd.DataFrame, target_col: str) -> dict:
    """
    รัน analysis ทั้งหมดแล้วคืน recommendations ครบชุด
    """
    return {
        "encoding":          analyze_encoding(df, target_col),
        "scaling":           analyze_scaling(df, target_col),
        "feature_selection": analyze_feature_selection(df, target_col),
    }