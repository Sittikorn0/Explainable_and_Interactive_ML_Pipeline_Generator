# Libraries
import datetime
import numpy as np
import pandas as pd

# Functions
def build_leaderboard_df(competition: dict) -> pd.DataFrame:
    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"], reverse=True,
    )
    errors = [(k, v) for k, v in competition.items() if v["cv_score"] is None]
    rows = []
    for i, (_, res) in enumerate(ranked):
        params = ", ".join(f"{k}={v}" for k, v in res["best_params"].items()) if res["best_params"] else "—"
        rows.append({"Rank": i + 1, "Model": res["label"],
                     "Cross-Val Score": res["cv_score"], "Cross-Val Std": res["cv_std"], "Hyperparameters": params})
    for _, res in errors:
        rows.append({"Rank": "—", "Model": res["label"],
                     "Cross-Val Score": None, "Cross-Val Std": None, "Hyperparameters": res.get("error", "")})
    return pd.DataFrame(rows)

def build_predictions_df(y_test, y_pred, task_type: str) -> pd.DataFrame:
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    if task_type == "classification":
        df["Correct"] = (y_test == y_pred)
    else:
        df["Error"] = y_pred - y_test
        df["Abs Error"] = np.abs(y_pred - y_test)
    return df