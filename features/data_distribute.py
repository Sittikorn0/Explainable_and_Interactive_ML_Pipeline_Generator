import os
from scipy.stats import skew
import numpy as np

def data_distribution(df):
    numeric_df = df.select_dtypes(include=['number']).dropna()
    total_outliers = 0
    outls_details = []

    for col in numeric_df.columns:
        col_skew = skew(numeric_df[col])
        
        # เลือก Method ตามความเบ้ของข้อมูล
        if abs(col_skew) < 0.5:
            # Z-Score (Normal Distribution)
            z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / numeric_df[col].std())
            count = (z_scores > 3).sum()
            method = "Z-Score"
        else:
            # IQR (Skewed Distribution)
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            count = ((numeric_df[col] < (Q1 - 1.5 * IQR)) | (numeric_df[col] > (Q3 + 1.5 * IQR))).sum()
            method = "IQR"
            
        total_outliers += count
        outls_details.append({
            "Column": col,
            "Outliers": count,
            "Method": method,
            "Skewness": round(col_skew, 2)
        })
        
    return total_outliers, outls_details