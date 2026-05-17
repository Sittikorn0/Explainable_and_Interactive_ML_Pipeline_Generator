import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import skew
from data_prepare.logic.data_type_detection import actual_type

def skew_insight(column_skew: float) -> str:
    """สร้างข้อความอธิบาย skewness"""
    absolute_skew = abs(column_skew)
    if absolute_skew < 0.5:
        shape_description = "ใกล้สมมาตร (Normal-like)"
    elif absolute_skew < 1:
        shape_description = "เบ้เล็กน้อย (Moderately Skewed)"
    else:
        shape_description = "เบ้มาก (Highly Skewed)"

    direction_description = ""
    if column_skew > 0.5:
        direction_description = " → หางยาวไปทางขวา"
    elif column_skew < -0.5:
        direction_description = " → หางยาวไปทางซ้าย"

    return f"**Skewness = {column_skew:.2f}** ({shape_description}{direction_description})"


def render_distributions_tab(dataframe: pd.DataFrame, target_column: str):
    st.subheader("Data Distributions")

    numeric_columns = dataframe.select_dtypes(include=["number"]).columns.tolist()
    datetime_columns = [
        column for column in dataframe.columns
        if pd.api.types.is_datetime64_any_dtype(dataframe[column])
        or (dataframe[column].dtype == object and actual_type(dataframe[column]) == "datetime")
    ]
    datetime_columns_set = set(datetime_columns)
    categorical_columns = [
        column for column in dataframe.select_dtypes(include=["object", "category"]).columns
        if column not in datetime_columns_set
    ]
    distribution_columns = numeric_columns + categorical_columns + datetime_columns

    if not distribution_columns:
        st.info("ไม่มีคอลัมน์ประเภท Numeric, Categorical หรือ Datetime สำหรับแสดง Distribution")
        return

    selected_distribution_column = st.selectbox(
        "Select column to visualize",
        distribution_columns,
        key="eda_dist_col",
    )

    if selected_distribution_column in numeric_columns:
        histogram_figure = px.histogram(
            dataframe,
            x=selected_distribution_column,
            marginal="box",
            nbins=30,
            color_discrete_sequence=["#7AA2F7"],
        )
        histogram_figure.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(histogram_figure, width="stretch")

        # Insight สำหรับ Numeric
        column_data = dataframe[selected_distribution_column].dropna()
        if len(column_data) < 2 or column_data.nunique() <= 1:
            st.info(f"**{selected_distribution_column}:** ข้อมูลไม่เพียงพอสำหรับวิเคราะห์รูปแบบการกระจายตัว")
        else:
            column_skew_value = float(skew(column_data))
            base_insight = skew_insight(column_skew_value)

            if abs(column_skew_value) >= 1:
                column_minimum = float(column_data.min())
                transformation_recommendation = (
                    "**Yeo-Johnson Transformation** (เนื่องจากมีค่า 0 หรือค่าลบ)"
                    if column_minimum <= 0 else
                    "**Log, Box-Cox, หรือ Yeo-Johnson Transformation**"
                )
                action_text = (
                    f"- **สถานะ:** ข้อมูลเบ้มาก (|Skew| ≥ 1)\n"
                    f"- **คำแนะนำ:** อาจต้องทำ Data Transformation ก่อนสร้างโมเดล เช่น {transformation_recommendation}"
                )
            elif abs(column_skew_value) >= 0.5:
                action_text = (
                    f"- **สถานะ:** ข้อมูลเบ้เล็กน้อย (0.5 ≤ |Skew| < 1)\n"
                    f"- **คำแนะนำ:** ยังสามารถใช้กับโมเดลส่วนใหญ่ได้โดยตรง แต่หากเบ้เข้าใกล้ 1 อาจพิจารณา Transform"
                )
            else:
                action_text = (
                    f"- **สถานะ:** ข้อมูลกระจายตัวแบบสมมาตร (|Skew| < 0.5)\n"
                    f"- **คำแนะนำ:** เหมาะสำหรับทุกโมเดล ไม่จำเป็นต้องจัดการเพิ่มเติม"
                )

            st.info(f"**ผลการวิเคราะห์: {selected_distribution_column}**\n\n{base_insight}\n\n{action_text}")

    elif selected_distribution_column in datetime_columns:
        # Datetime Distribution
        if pd.api.types.is_datetime64_any_dtype(dataframe[selected_distribution_column]):
            datetime_series = dataframe[selected_distribution_column].dropna()
        else:
            datetime_series = pd.to_datetime(
                dataframe[selected_distribution_column], format="mixed", dayfirst=False, errors="coerce"
            ).dropna()
        
        if len(datetime_series) == 0:
            st.info(f"**{selected_distribution_column}:** ไม่มีข้อมูล valid (ทั้งหมดเป็น NaN)")
        else:
            datetime_granularity = st.radio(
                "Granularity",
                ["Year", "Month", "Day"],
                horizontal=True,
                key="eda_dt_granularity",
            )
            if datetime_granularity == "Year":
                period_counts_dataframe = datetime_series.dt.year.value_counts().sort_index().reset_index()
                period_counts_dataframe.columns = [selected_distribution_column, "count"]
            elif datetime_granularity == "Month":
                period_counts_dataframe = (
                    datetime_series.dt.to_period("M")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                period_counts_dataframe.columns = [selected_distribution_column, "count"]
            else:
                period_counts_dataframe = (
                    datetime_series.dt.to_period("D")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .reset_index()
                )
                period_counts_dataframe.columns = [selected_distribution_column, "count"]

            line_chart_figure = px.line(
                period_counts_dataframe,
                x=selected_distribution_column,
                y="count",
                markers=True,
                color_discrete_sequence=["#7AA2F7"],
            )
            line_chart_figure.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(line_chart_figure, width="stretch")

            minimum_date = datetime_series.min().strftime("%Y-%m-%d")
            maximum_date = datetime_series.max().strftime("%Y-%m-%d")
            st.info(
                f"**{selected_distribution_column}:** ช่วงข้อมูลตั้งแต่ **{minimum_date}** ถึง **{maximum_date}** "
                f"— มีข้อมูลทั้งหมด **{len(datetime_series):,}** แถว"
            )

    else:
        # Categorical
        category_counts = dataframe[selected_distribution_column].value_counts().reset_index()
        category_counts.columns = [selected_distribution_column, "count"]
        bar_chart_figure = px.bar(
            category_counts.head(20),
            x=selected_distribution_column,
            y="count",
            color_discrete_sequence=["#7AA2F7"],
        )
        bar_chart_figure.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(bar_chart_figure, width="stretch")

        # Insight สำหรับ Categorical
        number_of_unique_categories = dataframe[selected_distribution_column].nunique()
        top_category_value = category_counts.iloc[0][selected_distribution_column] if len(category_counts) > 0 else "N/A"
        top_category_count = int(category_counts.iloc[0]["count"]) if len(category_counts) > 0 else 0
        top_category_percentage = (top_category_count / len(dataframe) * 100) if len(dataframe) > 0 else 0

        insight_lines = [
            f"**ผลการวิเคราะห์: {selected_distribution_column}**\n",
            f"- **จำนวนกลุ่ม (Unique):** {number_of_unique_categories} กลุ่ม",
            f"- **กลุ่มที่พบมากที่สุด:** **{top_category_value}** ({top_category_percentage:.1f}%)"
        ]

        # ตรวจ Class Imbalance (ถ้าเป็น Target)
        if selected_distribution_column == target_column and number_of_unique_categories <= 20:
            minimum_class_count = int(category_counts["count"].min())
            maximum_class_count = int(category_counts["count"].max())
            if maximum_class_count > 3 * minimum_class_count:
                imbalance_ratio = maximum_class_count / minimum_class_count
                insight_lines.append(f"\n[!] **แจ้งเตือน Class Imbalance:**")
                insight_lines.append(f"คลาสที่พบมากสุด มีจำนวนมากกว่าคลาสที่น้อยสุดถึง **{imbalance_ratio:.1f} เท่า**")
                insight_lines.append(f"*(อาจต้องจัดการข้อมูลก่อนสร้างโมเดล เช่น ทำ Oversampling, Undersampling, หรือปรับ Class Weight)*")

        st.info("\n".join(insight_lines))
