# Libraries
import pandas as pd
from scipy.stats import skew

# Logic Import
from backend.core.insight.reasoning_engine.engine import suggest

# Functions
def get_outlier_bounds(data_series: pd.Series) -> dict:
    """คำนวณขอบเขตของ Outlier โดยพิจารณาจาก Skewness ผ่าน Rule Engine"""
    clean_series = data_series.dropna()

    if len(clean_series) == 0 or clean_series.nunique() <= 1 or clean_series.std() == 0:
        return {
            "method": "N/A",
            "lower": float(data_series.min()),
            "upper": float(data_series.max()),
            "skewness": 0.0,
            "reason": "ค่าทุกแถวเหมือนกัน  ไม่สามารถตรวจจับ Outlier ได้"
        }

    if clean_series.empty:
        return {
            "method": "N/A",
            "lower": 0.0,
            "upper": 0.0,
            "skewness": 0.0,
            "reason": "ไม่มีข้อมูลเหลือหลังจากตัดค่าว่าง"
        }

    skewness_value = float(skew(clean_series))
    skewness_abs = abs(skewness_value)

    # ── ให้ Rule Engine ตัดสินใจวิธีตรวจจับ ──────────────────────
    rule_result = suggest("outlier_detection", {"skewness_abs": skewness_abs})
    method = rule_result["action"] if rule_result else ("iqr" if skewness_abs >= 0.5 else "zscore")
    rule_reason = rule_result["explanation"] if rule_result else ""

    if method == "zscore":
        mean_value = data_series.mean()
        std_value = data_series.std()
        return {
            "method": "Z-Score",
            "lower": mean_value - 3 * std_value,
            "upper": mean_value + 3 * std_value,
            "skewness": skewness_value,
            "reason": rule_reason or f"Skewness = {skewness_value:.2f} (ใกล้ 0 → กระจายแบบ Normal → เหมาะกับ Z-Score)",
            "rule_id": rule_result["rule_id"] if rule_result else "",
        }
    else:  # iqr
        quartile_1 = data_series.quantile(0.25)
        quartile_3 = data_series.quantile(0.75)
        interquartile_range = quartile_3 - quartile_1

        if interquartile_range == 0:
            return {
                "method": "N/A",
                "lower": float(quartile_1),
                "upper": float(quartile_3),
                "skewness": skewness_value,
                "reason": "ค่ากระจุกตัวมาก (IQR = 0)  ไม่สามารถตรวจจับ Outlier ด้วย IQR ได้",
                "rule_id": rule_result["rule_id"] if rule_result else "",
            }

        direction_text = "ขวา (Right-skewed)" if skewness_value > 0 else "ซ้าย (Left-skewed)"
        return {
            "method": "IQR",
            "lower": quartile_1 - 1.5 * interquartile_range,
            "upper": quartile_3 + 1.5 * interquartile_range,
            "skewness": skewness_value,
            "reason": rule_reason or f"Skewness = {skewness_value:.2f} (เบ้ไปทาง{direction_text} → เหมาะกับ IQR)",
            "rule_id": rule_result["rule_id"] if rule_result else "",
        }
