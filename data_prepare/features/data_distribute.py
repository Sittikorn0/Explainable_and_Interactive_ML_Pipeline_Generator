from scipy.stats import skew


def data_distribution(df):
    """ตรวจจับ Outlier ในคอลัมน์ตัวเลข โดยเลือกวิธีตาม Skewness ของข้อมูล"""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    total_outliers = 0
    outls_details = []

    for col in numeric_cols:
        # dropna แบบ per-column เพื่อไม่ให้ NA ของ column อื่นมากระทบ
        series = df[col].dropna()

        # ข้ามถ้าค่าเกือบเหมือนกันทั้งหมด — variance = 0 หมายความว่าไม่มี outlier จริงๆ
        # Z-Score: std=0 → division by zero, IQR: IQR=0 → ทุกค่าอยู่ในขอบเขต
        if series.nunique() <= 1 or series.std() == 0:
            outls_details.append({
                "Column": col,
                "Outliers": 0,
                "Method": "N/A",
                "Skewness": 0.0,
                "Lower": float(series.min()),
                "Upper": float(series.max()),
                "Reason": "ค่าทุกแถวเหมือนกัน — ไม่สามารถตรวจจับ Outlier ได้",
            })
            continue

        col_skew = skew(series)

        # เลือก Method ตามความเบ้ของข้อมูล (ref: Topic 8 - Outlier Detection)
        # |skewness| < 0.5 → ข้อมูลใกล้ Normal → ใช้ Z-Score
        # |skewness| >= 0.5 → ข้อมูล Skewed → ใช้ IQR ซึ่งทนต่อ Outlier ได้ดีกว่า
        if abs(col_skew) < 0.5:
            mean = series.mean()
            std = series.std()
            lower = mean - 3 * std
            upper = mean + 3 * std
            count = int(((series < lower) | (series > upper)).sum())
            method = "Z-Score"
            reason = f"Skewness = {col_skew:.2f} (ใกล้ 0 → กระจายแบบ Normal → เหมาะกับ Z-Score)"
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            # IQR = 0 หมายความว่าค่ากระจุกตัวมาก เช่น binary column ที่ค่าหนึ่งมี > 75%
            # ถ้าคำนวณต่อ lower = upper = Q1 → ค่าอื่นทั้งหมดกลายเป็น outlier ซึ่งผิดหลักการ
            if iqr == 0:
                outls_details.append({
                    "Column": col,
                    "Outliers": 0,
                    "Method": "N/A",
                    "Skewness": round(col_skew, 2),
                    "Lower": float(q1),
                    "Upper": float(q3),
                    "Reason": "ค่ากระจุกตัวมาก (IQR = 0) — ไม่สามารถตรวจจับ Outlier ด้วย IQR ได้",
                })
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            count = int(((series < lower) | (series > upper)).sum())
            method = "IQR"
            direction = "ขวา (Right-skewed)" if col_skew > 0 else "ซ้าย (Left-skewed)"
            reason = f"Skewness = {col_skew:.2f} (เบ้ไปทาง{direction} → เหมาะกับ IQR)"

        total_outliers += count
        outls_details.append({
            "Column": col,
            "Outliers": count,
            "Method": method,
            "Skewness": round(col_skew, 2),
            "Lower": lower,
            "Upper": upper,
            "Reason": reason,
        })

    return total_outliers, outls_details