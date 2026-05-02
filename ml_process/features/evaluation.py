"""ml_process/evaluation.py — metrics + visualizations"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix,
)


def get_metrics(y_test, y_pred, task_type: str) -> dict:
    if task_type == "classification":
        # macro: ค่าเฉลี่ยแบบไม่ถ่วงน้ำหนัก ให้ทุก class มีความสำคัญเท่ากัน
        # หมายเหตุ: weighted recall = accuracy เสมอ (ทางคณิตศาสตร์)
        #           จึงใช้ macro เพื่อให้ค่าทั้ง 4 แสดงผลที่แตกต่างและมีความหมาย
        return {
            "Accuracy":       round(accuracy_score(y_test, y_pred), 4),
            "Precision(Mac)": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "Recall(Mac)":    round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
            "F1(Mac)":        round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        }
    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE":      round(mse, 4),
        "RMSE":     round(float(np.sqrt(mse)), 4),
        "R² Score": round(r2_score(y_test, y_pred), 4),
    }


def show_metrics(metrics: dict):
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(k, v)


def show_leaderboard(competition: dict):
    ranked = sorted([(k, v) for k, v in competition.items() if v["cv_score"] is not None],
                    key=lambda x: x[1]["cv_score"], reverse=True)
    errors = [(k, v) for k, v in competition.items() if v["cv_score"] is None]

    medals = ["#1", "#2", "#3"]
    rows = []
    for i, (_, res) in enumerate(ranked):
        params = " | ".join(f"{k}={v}" for k, v in res["best_params"].items()) if res["best_params"] else "—"
        rows.append({
            "Rank": medals[i] if i < 3 else str(i + 1),
            "Model": res["label"],
            "CV Score": res["cv_score"],
            "±Std": res["cv_std"],
            "Best Params": params,
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "CV Score": st.column_config.NumberColumn(format="%.4f"),
            "±Std":     st.column_config.NumberColumn(format="%.4f"),
        },
    )
    if errors:
        with st.expander(f"{len(errors)} model ที่ข้ามไป"):
            for _, res in errors:
                st.caption(f"**{res['label']}**: {res['error']}")


def show_confusion_matrix(y_test, y_pred):
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred).flatten()
    labels = sorted(list(set(y_test) | set(y_pred)), key=str)
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    fig    = px.imshow(cm, x=[str(l) for l in labels], y=[str(l) for l in labels],
                       text_auto=True, color_continuous_scale="Blues",
                       labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")


def show_pred_vs_actual(y_test, y_pred):
    df_plot = pd.DataFrame({"Actual": np.array(y_test).flatten(),
                            "Predicted": np.array(y_pred).flatten()})
    min_v, max_v = min(df_plot.min()), max(df_plot.max())
    fig = px.scatter(df_plot, x="Actual", y="Predicted", opacity=0.6,
                     color_discrete_sequence=["#58a6ff"])
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v,
                  line=dict(color="red", dash="dash", width=1.5))
    fig.update_layout(template="plotly_dark", height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")