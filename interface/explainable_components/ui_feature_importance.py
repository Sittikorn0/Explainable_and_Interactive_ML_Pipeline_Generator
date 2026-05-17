import streamlit as st
import plotly.graph_objects as go
from explainable.logic.explainer import compute_permutation_importance
from interface.explainable_components.utils import (
    render_section_header, BACKGROUND_COLOR, BORDER_COLOR, TEXT_DIM_COLOR, BORDER_RADIUS
)

def render_importance(model, X_test, y_test, task_type):
    render_section_header(
        "Feature ไหนสำคัญที่สุด?",
        "ระบบสลับค่าของแต่ละ feature แล้วดูว่า model แย่ลงแค่ไหน — แท่งยาว = สำคัญมาก",
    )

    cache_key = f"_xai_perm_{id(model)}"
    if st.session_state.get(cache_key) is None:
        with st.spinner("กำลังวิเคราะห์ความสำคัญของ feature..."):
            permutation_dataframe = compute_permutation_importance(model, X_test, y_test, task_type)
            st.session_state[cache_key] = permutation_dataframe
    permutation_dataframe = st.session_state[cache_key]

    top_features_count   = min(15, len(permutation_dataframe))
    plot_dataframe = permutation_dataframe.head(top_features_count)

    bar_chart_figure = go.Figure(go.Bar(
        x=plot_dataframe["Importance"],
        y=plot_dataframe["Feature"],
        orientation="h",
        marker=dict(color=plot_dataframe["Importance"].tolist(), colorscale=[[0, "rgba(122, 162, 247, 0.1)"], [1, "#7AA2F7"]], showscale=False),
        text=plot_dataframe["Importance"].round(3).tolist(),
        textposition="outside",
    ))
    bar_chart_figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=max(320, top_features_count * 36),
        yaxis=dict(autorange="reversed"),
        xaxis_title="ความสำคัญ (ยิ่งสูง = ยิ่งสำคัญ)",
        margin=dict(t=10, b=30, l=10, r=90),
    )
    st.plotly_chart(bar_chart_figure, width="stretch")

    positive_importance_features = permutation_dataframe[permutation_dataframe["Importance"] > 0]
    if len(positive_importance_features) == 0:
        st.warning("ทุก feature มีความสำคัญน้อยมาก — ลองตรวจสอบ dataset อีกครั้ง")
        return

    render_section_header("3 Feature สำคัญที่สุด")

    rank_colors  = ["#E0AF68", "#BB9AF7", "#7AA2F7"]
    rank_labels  = ["#1", "#2", "#3"]
    total_importance = positive_importance_features["Importance"].sum() + 1e-9
    top_3_features = positive_importance_features.head(3).reset_index(drop=True)
    card_columns = st.columns(len(top_3_features))

    for index, col in enumerate(card_columns):
        row_data = top_3_features.iloc[index]
        percentage = row_data["Importance"] / total_importance * 100
        with col:
            st.markdown(
                f'<div style="background:{BACKGROUND_COLOR};border:1px solid {BORDER_COLOR};'
                f'border-top:3px solid {rank_colors[index]};border-radius:{BORDER_RADIUS};'
                f'padding:20px 16px;text-align:center">'
                f'<div style="color:{rank_colors[index]};font-size:0.85rem;font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:10px">{rank_labels[index]}</div>'
                f'<div style="font-family:monospace;color:#e6edf3;font-size:1rem;'
                f'word-break:break-all;margin-bottom:12px">{row_data["Feature"]}</div>'
                f'<div style="color:{rank_colors[index]};font-size:1.6rem;font-weight:800;'
                f'line-height:1">{percentage:.0f}%</div>'
                f'<div style="color:{TEXT_DIM_COLOR};font-size:0.85rem;margin-top:6px">'
                f'ของความสำคัญรวม</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    useless_features = permutation_dataframe[permutation_dataframe["Importance"] <= 0]
    if len(useless_features):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            f"**Feature ที่แทบไม่มีผล ({len(useless_features)} ตัว):**\n"
            + ", ".join(f"`{feature}`" for feature in useless_features["Feature"].head(5))
            + (" ..." if len(useless_features) > 5 else "")
        )
