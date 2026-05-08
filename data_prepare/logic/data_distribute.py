from data_prepare.logic.statistics import get_outlier_bounds

def data_distribution(dataset, fixed_bounds=None):
    """ตรวจจับ Outlier ในคอลัมน์ตัวเลข โดยใช้ statistics helper"""
    numeric_columns = dataset.select_dtypes(include=["number"]).columns
    total_outliers_count = 0
    outlier_details = []

    for column_name in numeric_columns:
        data_series = dataset[column_name].dropna()
        if len(data_series) == 0:
            continue
            
        if fixed_bounds and column_name in fixed_bounds:
            bounds = fixed_bounds[column_name]
        else:
            bounds = get_outlier_bounds(data_series)
            
        lower_bound = bounds["lower"]
        upper_bound = bounds["upper"]
        
        if bounds["method"] == "N/A":
            outlier_count = 0
        else:
            outlier_count = int(((data_series < lower_bound) | (data_series > upper_bound)).sum())

        total_outliers_count += outlier_count
        outlier_details.append({
            "Column": column_name,
            "Outliers": outlier_count,
            "Method": bounds["method"],
            "Skewness": round(bounds["skewness"], 2),
            "Lower": float(lower_bound),
            "Upper": float(upper_bound),
            "Reason": bounds["reason"],
        })

    return total_outliers_count, outlier_details