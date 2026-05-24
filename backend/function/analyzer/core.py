# Libraries
import pandas as pd
import numpy as np
from scipy.stats import skew

# Logic Import
from backend.function.analyzer.leakage import analyze_leakage
from backend.function.analyzer.encoding import analyze_encoding
from backend.function.analyzer.scaling import analyze_scaling
from backend.function.analyzer.feature_selection import analyze_feature_selection

# Functions
# รัน Analysis ทั้งหมด (Encoding/Scaling/Feature Selection/Leakage) คืน dict ใช้ใน suggestion_engine และ data_overview_page
def analyze_all(dataset: pd.DataFrame, target_column: str) -> dict:
    return {
        "encoding": analyze_encoding(dataset, target_column),
        "scaling": analyze_scaling(dataset, target_column),
        "feature_selection": analyze_feature_selection(dataset, target_column),
        "leakage": analyze_leakage(dataset, target_column),
    }