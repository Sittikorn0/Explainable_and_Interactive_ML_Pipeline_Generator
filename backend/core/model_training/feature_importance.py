# Libraries
import numpy as np
import pandas as pd

# Logic Import
from backend.core.model_training.preprocess.pipeline import preprocess
from backend.core.model_training.trainer.train_model import get_model_map

# Functions
# คำนวณ Feature Importance จาก feature_importances_ หรือ coef_ ของ best model คืน (fi_df, error_msg) ใช้ใน model_process_page
def compute_fi(df: pd.DataFrame, target_col: str,
               best_key: str, best_params: dict,
               trans_summary: dict) -> tuple[pd.DataFrame | None, str | None]:

    try:
        scaling_method     = trans_summary.get("scaling_method", "no_scaling")
        encoding_decisions = trans_summary.get("encoding_decisions") or None
        X_tr, _, y_tr, _, _ = preprocess(
            df, target_col,
            scaling_method=scaling_method,
            encoding_decisions=encoding_decisions,
        )
        m = get_model_map()[best_key]()
        if best_params:
            try:
                m.set_params(**best_params)
            except Exception:
                pass
        m.fit(X_tr, y_tr)

        importances = None
        if hasattr(m, "feature_importances_"):
            importances = m.feature_importances_
        elif hasattr(m, "coef_"):
            coef = m.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        if importances is None:
            return None, None

        fi_df = (pd.DataFrame({"Feature": X_tr.columns, "Importance": importances})
                   .sort_values("Importance", ascending=False)
                   .reset_index(drop=True))
        return fi_df, None

    except Exception as e:
        return None, str(e)

