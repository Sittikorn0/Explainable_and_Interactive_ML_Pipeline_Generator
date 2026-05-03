import pandas as pd
from scipy.stats import skew

def get_outlier_bounds(series: pd.Series) -> dict:
    """คำนวณขอบเขตของ Outlier โดยพิจารณาจาก Skewness อัตโนมัติ"""
    if series.nunique() <= 1 or series.std() == 0:
        return {
            "method": "N/A",
            "lower": float(series.min()),
            "upper": float(series.max()),
            "skewness": 0.0,
            "reason": "ค่าทุกแถวเหมือนกัน — ไม่สามารถตรวจจับ Outlier ได้"
        }
        
    col_skew = float(skew(series))
    
    if abs(col_skew) < 0.5:
        mean = series.mean()
        std = series.std()
        return {
            "method": "Z-Score",
            "lower": mean - 3 * std,
            "upper": mean + 3 * std,
            "skewness": col_skew,
            "reason": f"Skewness = {col_skew:.2f} (ใกล้ 0 → กระจายแบบ Normal → เหมาะกับ Z-Score)"
        }
    else:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:
            return {
                "method": "N/A",
                "lower": float(q1),
                "upper": float(q3),
                "skewness": col_skew,
                "reason": "ค่ากระจุกตัวมาก (IQR = 0) — ไม่สามารถตรวจจับ Outlier ด้วย IQR ได้"
            }
            
        direction = "ขวา (Right-skewed)" if col_skew > 0 else "ซ้าย (Left-skewed)"
        return {
            "method": "IQR",
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
            "skewness": col_skew,
            "reason": f"Skewness = {col_skew:.2f} (เบ้ไปทาง{direction} → เหมาะกับ IQR)"
        }
