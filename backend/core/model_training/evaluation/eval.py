# Libraries
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, confusion_matrix,
)

# UI Import
from user_interface.pages.Model_process.model_components.model_compo import render_metric_cards

# Function
# คำนวณ metrics จาก y_test/y_pred ตาม task type (Accuracy/F1 สำหรับ clf, MSE/R² สำหรับ reg) ใช้ใน model_process_page
def get_metrics(actual_values, predicted_values, task_type: str) -> dict:
    if task_type == "classification":
        actual_values = pd.Series(actual_values).astype(str)
        predicted_values = pd.Series(predicted_values).astype(str)
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

# render metric cards ใน UI โดยส่งต่อให้ render_metric_cards ใช้ใน model_process_page
def show_metrics(evaluation_metrics: dict):
    render_metric_cards(evaluation_metrics)

# แสดง residual scatter plot (Predicted vs Error) ใช้ตรวจ systematic bias สำหรับ Regression
def show_residual_plot(actual_values, predicted_values):
    actual = np.array(actual_values).flatten()
    predicted = np.array(predicted_values).flatten()
    residuals = actual - predicted
    
    fig = px.scatter(
        x=predicted, y=residuals, opacity=0.6,
        labels={'x': 'Predicted Values', 'y': 'Residuals (Actual - Predicted)'},
        color_discrete_sequence=["#F59E0B"]
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#EF4444")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

# แสดง histogram การกระจายของ error ใช้ดู bias ของ Regression model
def show_error_dist(actual_values, predicted_values):
    actual = np.array(actual_values).flatten()
    predicted = np.array(predicted_values).flatten()
    errors = actual - predicted
    
    fig = px.histogram(
        errors, nbins=50, 
        labels={'value': 'Prediction Error'},
        color_discrete_sequence=["#BB9AF7"]
    )
    fig.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig, width="stretch")

# แสดงตาราง leaderboard จัดอันดับ model ตาม CV score ใช้ใน model_process_page
def show_leaderboard(model_competition_results: dict):
    ranked_models = sorted(
        [(model_name, result) for model_name, result in model_competition_results.items() if result["cv_score"] is not None],
        key=lambda item: item[1].get("cv_score_raw", item[1]["cv_score"]), reverse=True
    )
    failed_models = [(model_name, result) for model_name, result in model_competition_results.items() if result["cv_score"] is None]

    medals = ["#1", "#2", "#3"]
    leaderboard_rows = []
    
    for index, (_, result) in enumerate(ranked_models):
        params = result["best_params"] or {}
        # Prettify keys: learning_rate -> Learning Rate
        param_list = [f"{k.replace('_', ' ').title()}: {v}" for k, v in params.items()]
        parameters_text = ", ".join(param_list) if param_list else ""
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
            "Rank": st.column_config.TextColumn(
                "Rank",
                help="อันดับของโมเดล (วัดจากค่าเฉลี่ยคะแนนประสิทธิภาพ)"
            ),
            "Model": st.column_config.TextColumn(
                "Model",
                help="ชื่อโมเดล Machine Learning"
            ),
            "CV Score": st.column_config.NumberColumn(
                "Cross-Val Score",
                help="คะแนนประสิทธิภาพเฉลี่ยจากการทำ Cross-Validation (ยิ่งสูงยิ่งดี)",
                format="%.4f"
            ),
            "±Std": st.column_config.NumberColumn(
                "Cross-Val Std",
                help="ความผันผวนของคะแนน (ยิ่งต่ำยิ่งเสถียรและเชื่อถือได้)",
                format="%.4f"
            ),
            "Best Params": st.column_config.TextColumn(
                "Hyperparameters",
                help="การตั้งค่าพารามิเตอร์ที่เหมาะสมที่สุดสำหรับโมเดลนี้"
            ),
        },
    )
    
    if failed_models:
        with st.expander(f"{len(failed_models)} model ที่ข้ามไป"):
            for _, result in failed_models:
                st.caption(f"**{result['label']}**: {result['error']}")

# แสดง confusion matrix สำหรับ Classification ใช้ใน model_process_page
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
    
    # ปรับแต่ง UI กราฟให้ดูพรีเมียมขึ้น
    figure.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(t=40, b=40, l=40, r=40),
        font=dict(size=13),
        xaxis_title=dict(font=dict(size=14, color="#94A3B8")),
        yaxis_title=dict(font=dict(size=14, color="#94A3B8")),
        coloraxis_showscale=True # นำแถบสีกลับมาตามต้องการ
    )
    
    # ขยายตัวเลขในช่องตาราง
    figure.update_traces(
        textfont_size=16,
        texttemplate="%{z}" # แสดงเฉพาะตัวเลขจำนวน
    )
    
    st.plotly_chart(figure, width="stretch")

# แสดง scatter plot Predicted vs Actual ใช้ตรวจ fit ของ Regression model ใช้ใน model_process_page
def show_pred_vs_actual(actual_values, predicted_values):
    plot_dataset = pd.DataFrame({
        "Actual": np.array(actual_values).flatten(),
        "Predicted": np.array(predicted_values).flatten()
    })
    
    min_value = min(plot_dataset.min())
    max_value = max(plot_dataset.max())
    
    figure = px.scatter(
        plot_dataset, x="Actual", y="Predicted", opacity=0.6,
        color_discrete_sequence=["#7AA2F7"]
    )
    figure.add_shape(
        type="line", x0=min_value, y0=min_value, x1=max_value, y1=max_value,
        line=dict(color="#EF4444", dash="dash", width=1.5)
    )
    figure.update_layout(template="plotly_dark", height=400, margin=dict(t=20, b=20))
    st.plotly_chart(figure, width="stretch")