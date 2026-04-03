from scipy.stats import skew
import numpy as np


def data_distribution(df):
    """ตรวจจับ Outlier ในคอลัมน์ตัวเลข โดยเลือกวิธีตาม Skewness ของข้อมูล"""
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    total_outliers = 0
    outls_details = []

    for col in numeric_df.columns:
        col_skew = skew(numeric_df[col])

        # เลือก Method ตามความเบ้ของข้อมูล (ref: Topic 8 - Outlier Detection)
        # |skewness| < 0.5 → ข้อมูลใกล้ Normal → ใช้ Z-Score
        # |skewness| >= 0.5 → ข้อมูล Skewed → ใช้ IQR ซึ่งทนต่อ Outlier ได้ดีกว่า
        if abs(col_skew) < 0.5:
            mean = numeric_df[col].mean()
            std = numeric_df[col].std()
            z_scores = np.abs((numeric_df[col] - mean) / std)
            count = int((z_scores > 3).sum())
            lower = mean - 3 * std
            upper = mean + 3 * std
            method = "Z-Score"
            reason = f"Skewness = {col_skew:.2f} (ใกล้ 0 → กระจายแบบ Normal → เหมาะกับ Z-Score)"
        else:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((numeric_df[col] < lower) | (numeric_df[col] > upper)).sum())
            method = "IQR"
            direction = "ขวา (Right-skewed)" if col_skew > 0 else "ซ้าย (Left-skewed)"
            reason = f"Skewness = {col_skew:.2f} (เบ้ไปทาง{direction} → เหมาะกับ IQR)"

        total_outliers += count
        outls_details.append({
            "Column": col,
            "Outliers": count,
            "Method": method,
            "Skewness": round(col_skew, 2),
            "Lower": round(lower, 4),
            "Upper": round(upper, 4),
            "Reason": reason,
        })

    return total_outliers, outls_details