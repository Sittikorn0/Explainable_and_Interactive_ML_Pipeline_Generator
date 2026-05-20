# Libraries
import pandas as pd
from scipy.stats import skew

# Functions
def get_outlier_bounds(data_series: pd.Series) -> dict:
    """คำนวณขอบเขตของ Outlier โดยพิจารณาจาก Skewness อัตโนมัติ"""
    clean_series = data_series.dropna()
    
    if len(clean_series) == 0 or clean_series.nunique() <= 1 or clean_series.std() == 0:
        return {
            "method": "N/A",
            "lower": float(data_series.min()),
            "upper": float(data_series.max()),
            "skewness": 0.0,
            "reason": "ค่าทุกแถวเหมือนกัน  ไม่สามารถตรวจจับ Outlier ได้"
        }
        
    # ต้อง dropna() ก่อนคำนวณ skewness เสมอ ไม่เช่นนั้นจะรีเทิร์น NaN ถ้ามีค่าว่าง
    if clean_series.empty:
        return {
            "method": "N/A",
            "lower": 0.0,
            "upper": 0.0,
            "skewness": 0.0,
            "reason": "ไม่มีข้อมูลเหลือหลังจากตัดค่าว่าง"
        }
        
    skewness_value = float(skew(clean_series))
    
    if abs(skewness_value) < 0.5:
        mean_value = data_series.mean()
        std_value = data_series.std()
        return {
            "method": "Z-Score",
            "lower": mean_value - 3 * std_value,
            "upper": mean_value + 3 * std_value,
            "skewness": skewness_value,
            "reason": f"Skewness = {skewness_value:.2f} (ใกล้ 0 → กระจายแบบ Normal → เหมาะกับ Z-Score)"
        }
    else:
        quartile_1 = data_series.quantile(0.25)
        quartile_3 = data_series.quantile(0.75)
        interquartile_range = quartile_3 - quartile_1
        
        if interquartile_range == 0:
            return {
                "method": "N/A",
                "lower": float(quartile_1),
                "upper": float(quartile_3),
                "skewness": skewness_value,
                "reason": "ค่ากระจุกตัวมาก (IQR = 0)  ไม่สามารถตรวจจับ Outlier ด้วย IQR ได้"
            }
            
        direction_text = "ขวา (Right-skewed)" if skewness_value > 0 else "ซ้าย (Left-skewed)"
        return {
            "method": "IQR",
            "lower": quartile_1 - 1.5 * interquartile_range,
            "upper": quartile_3 + 1.5 * interquartile_range,
            "skewness": skewness_value,
            "reason": f"Skewness = {skewness_value:.2f} (เบ้ไปทาง{direction_text} → เหมาะกับ IQR)"
        }
