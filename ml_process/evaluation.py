"""
ml_process/evaluate.py
Metrics calculation + Visualization components
"""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix,
)


# ── Metrics ───────────────────────────────────────────────────
def get_metrics(y_test, y_pred, task_type: str) -> dict:
    """
    คำนวณ metrics ตาม task type

    Classification → Accuracy, Precision, Recall, F1-Score
    Regression     → MSE, RMSE, R² Score
    """
    if task_type == "classification":
        avg = "weighted"
        return {
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4),
            "F1-Score":  round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4),
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        return {
            "MSE":      round(mse, 4),
            "RMSE":     round(float(np.sqrt(mse)), 4),
            "R² Score": round(r2_score(y_test, y_pred), 4),
        }


# ── UI Components ─────────────────────────────────────────────
def show_leaderboard(competition: dict):
    """
    ตาราง leaderboard เรียงตาม CV score จากสูงสุด
    แสดง medal 🥇🥈🥉 และ best params
    """
    import streamlit as st
    ranked = sorted(
        [(k, v) for k, v in competition.items() if v["cv_score"] is not None],
        key=lambda x: x[1]["cv_score"],
        reverse=True,
    )
    errors = [(k, v) for k, v in competition.items() if v["cv_score"] is None]

    medals = ["🥇", "🥈", "🥉"]
    rows   = []
    for i, (key, res) in enumerate(ranked):
        params_str = (
            " | ".join(f"{k}={v}" for k, v in res["best_params"].items())
            if res["best_params"] else "—"
        )
        rows.append({
            "Rank":        medals[i] if i < 3 else str(i + 1),
            "Model":       res["label"],
            "CV Score":    res["cv_score"],
            "±Std":        res["cv_std"],
            "Best Params": params_str,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank":        st.column_config.TextColumn("Rank",      width="small"),
            "Model":       st.column_config.TextColumn("Model",     width="medium"),
            "CV Score":    st.column_config.NumberColumn("CV Score", format="%.4f"),
            "±Std":        st.column_config.NumberColumn("±Std",     format="%.4f"),
            "Best Params": st.column_config.TextColumn("Best Params"),
        },
    )

    if errors:
        with st.expander(f"⚠ {len(errors)} model ที่ข้ามไป"):
            for _, res in errors:
                st.caption(f"**{res['label']}**: {res['error']}")


def show_metrics(metrics: dict):
    """Metric cards แสดงแบบ columns"""
    import streamlit as st
    cols = st.columns(len(metrics))
    for i, (name, val) in enumerate(metrics.items()):
        cols[i].metric(name, val)


def show_confusion_matrix(y_test, y_pred):
    """Plotly heatmap สำหรับ confusion matrix"""
    import streamlit as st
    import plotly.express as px
    labels = sorted(list(set(y_test) | set(y_pred)), key=str)
    cm     = confusion_matrix(y_test, y_pred, labels=labels)

    fig = px.imshow(
        cm,
        x=[str(l) for l in labels],
        y=[str(l) for l in labels],
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        template="plotly_dark",
        height=420,
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_pred_vs_actual(y_test, y_pred):
    """Scatter plot Actual vs Predicted สำหรับ regression"""
    import streamlit as st
    import plotly.express as px
    y_test_arr = np.array(y_test).flatten()
    y_pred_arr = np.array(y_pred).flatten()

    df_plot = pd.DataFrame({"Actual": y_test_arr, "Predicted": y_pred_arr})
    min_v   = min(df_plot["Actual"].min(), df_plot["Predicted"].min())
    max_v   = max(df_plot["Actual"].max(), df_plot["Predicted"].max())

    fig = px.scatter(
        df_plot, x="Actual", y="Predicted",
        opacity=0.6, color_discrete_sequence=["#0082CE"],
        trendline="ols",
    )
    # เส้น perfect prediction
    fig.add_shape(
        type="line",
        x0=min_v, y0=min_v, x1=max_v, y1=max_v,
        line=dict(color="red", dash="dash", width=1.5),
    )
    fig.update_layout(template="plotly_dark", height=420)
    st.plotly_chart(fig, use_container_width=True)