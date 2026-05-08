"""ml_process/features/evaluation.py — metrics + visualizations"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix,
)

def get_metrics(actual_values, predicted_values, task_type: str) -> dict:
    if task_type == "classification":
        # macro: ค่าเฉลี่ยแบบไม่ถ่วงน้ำหนัก ให้ทุก class มีความสำคัญเท่ากัน
        # หมายเหตุ: weighted recall = accuracy เสมอ (ทางคณิตศาสตร์)
        #           จึงใช้ macro เพื่อให้ค่าทั้ง 4 แสดงผลที่แตกต่างและมีความหมาย
        return {
            "Accuracy":       round(accuracy_score(actual_values, predicted_values), 4),
            "Precision(Mac)": round(precision_score(actual_values, predicted_values, average="macro", zero_division=0), 4),
            "Recall(Mac)":    round(recall_score(actual_values, predicted_values, average="macro", zero_division=0), 4),
            "F1(Mac)":        round(f1_score(actual_values, predicted_values, average="macro", zero_division=0), 4),
        }
        
    mse_score = mean_squared_error(actual_values, predicted_values)
    return {
        "MSE":      round(mse_score, 4),
        "RMSE":     round(float(np.sqrt(mse_score)), 4),
        "R² Score": round(r2_score(actual_values, predicted_values), 4),
    }

def show_metrics(evaluation_metrics: dict):
    columns = st.columns(len(evaluation_metrics))
    for index, (metric_name, metric_value) in enumerate(evaluation_metrics.items()):
        columns[index].metric(metric_name, metric_value)

def show_leaderboard(model_competition_results: dict):
    ranked_models = sorted(
        [(model_name, result) for model_name, result in model_competition_results.items() if result["cv_score"] is not None],
        key=lambda item: item[1]["cv_score"], reverse=True
    )
    failed_models = [(model_name, result) for model_name, result in model_competition_results.items() if result["cv_score"] is None]

    medals = ["#1", "#2", "#3"]
    leaderboard_rows = []
    
    for index, (_, result) in enumerate(ranked_models):
        parameters_text = " | ".join(f"{param_key}={param_value}" for param_key, param_value in result["best_params"].items()) if result["best_params"] else "—"
        leaderboard_rows.append({
            "Rank": medals[index] if index < 3 else str(index + 1),
            "Model": result["label"],
            "CV Score": result["cv_score"],
            "±Std": result["cv_std"],
            "Best Params": parameters_text,
        })

    st.dataframe(
        pd.DataFrame(leaderboard_rows),
        hide_index=True,
        width="stretch",
        column_config={
            "CV Score": st.column_config.NumberColumn(format="%.4f"),
            "±Std":     st.column_config.NumberColumn(format="%.4f"),
        },
    )
    
    if failed_models:
        with st.expander(f"{len(failed_models)} model ที่ข้ามไป"):
            for _, result in failed_models:
                st.caption(f"**{result['label']}**: {result['error']}")

def show_confusion_matrix(actual_values, predicted_values):
    actual_array = np.array(actual_values).flatten()
    predicted_array = np.array(predicted_values).flatten()
    
    unique_labels = sorted(list(set(actual_array) | set(predicted_array)), key=str)
    confusion_mat = confusion_matrix(actual_array, predicted_array, labels=unique_labels)
    
    figure = px.imshow(
        confusion_mat, 
        x=[str(label) for label in unique_labels], 
        y=[str(label) for label in unique_labels],
        text_auto=True, 
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count")
    )
    figure.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
    st.plotly_chart(figure, width="stretch")

def show_pred_vs_actual(actual_values, predicted_values):
    plot_dataset = pd.DataFrame({
        "Actual": np.array(actual_values).flatten(),
        "Predicted": np.array(predicted_values).flatten()
    })
    
    min_value = min(plot_dataset.min())
    max_value = max(plot_dataset.max())
    
    figure = px.scatter(
        plot_dataset, x="Actual", y="Predicted", opacity=0.6,
        color_discrete_sequence=["#58a6ff"]
    )
    figure.add_shape(
        type="line", x0=min_value, y0=min_value, x1=max_value, y1=max_value,
        line=dict(color="red", dash="dash", width=1.5)
    )
    figure.update_layout(template="plotly_dark", height=420, margin=dict(t=20, b=20))
    st.plotly_chart(figure, width="stretch")