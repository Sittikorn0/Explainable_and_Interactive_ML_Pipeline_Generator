from data_prepare.features.statistics import get_outlier_bounds

def data_distribution(df):
    """ตรวจจับ Outlier ในคอลัมน์ตัวเลข โดยใช้ statistics helper"""
    numeric_cols = df.select_dtypes(include=["number"]).columns
    total_outliers = 0
    outls_details = []

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        bounds = get_outlier_bounds(series)
        lower = bounds["lower"]
        upper = bounds["upper"]
        
        if bounds["method"] == "N/A":
            count = 0
        else:
            count = int(((series < lower) | (series > upper)).sum())

        total_outliers += count
        outls_details.append({
            "Column": col,
            "Outliers": count,
            "Method": bounds["method"],
            "Skewness": round(bounds["skewness"], 2),
            "Lower": float(lower),
            "Upper": float(upper),
            "Reason": bounds["reason"],
        })

    return total_outliers, outls_details