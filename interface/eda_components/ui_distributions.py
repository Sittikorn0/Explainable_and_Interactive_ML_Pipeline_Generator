import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import skew
from data_prepare.logic.data_type_detection import actual_type

def skew_insight(column_skew: float) -> str:
    """สร้างข้อความอธิบาย skewness"""
    absolute_skew = abs(column_skew)
    if absolute_skew < 0.5:
        shape_description = "ใกล้ Normal (สมมาตร)"
    elif absolute_skew < 1:
        shape_description = "เบ้เล็กน้อย (Moderately Skewed)"
    else:
        shape_description = "เบ้มาก (Highly Skewed)"

    direction_description = ""
    if column_skew > 0.5:
        direction_description = " → หางยาวไปทางขวา (Right-skewed)"
    elif column_skew < -0.5:
        direction_description = " → หางยาวไปทางซ้าย (Left-skewed)"

    return f"Skewness = {column_skew:.2f} → {shape_description}{direction_description}"


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
            color_discrete_sequence=["#0082CE"],
        )
        histogram_figure.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(histogram_figure, width="stretch")

        # Insight สำหรับ Numeric
        column_data = dataframe[selected_distribution_column].dropna()
        if len(column_data) < 2 or column_data.nunique() <= 1:
            st.info(f"**{selected_distribution_column}:** ข้อมูลไม่เพียงพอสำหรับวิเคราะห์ Skewness")
        else:
            column_skew_value = float(skew(column_data))
            skewness_insight = skew_insight(column_skew_value)

            if abs(column_skew_value) >= 1:
                column_minimum = float(column_data.min())
                if column_minimum <= 0:
                    transformation_recommendation = "**Yeo-Johnson Transformation** (รองรับค่า 0 และค่าลบ)"
                else:
                    transformation_recommendation = "Log, Box-Cox, หรือ Yeo-Johnson Transformation"
                skewness_insight += (
                    f"\n\n**คำแนะนำ** (อ้างอิง Topic 9): ข้อมูลเบ้มาก (|Skew| ≥ 1) "
                    f"อาจต้อง Transform ก่อนสร้างโมเดล เช่น {transformation_recommendation}"
                )
            elif abs(column_skew_value) >= 0.5:
                skewness_insight += (
                    "\n\n**เกณฑ์การตัดสิน** (อ้างอิง Topic 2): "
                    "|Skew| ≥ 0.5 = เบ้เล็กน้อย — ยังใช้ได้กับโมเดลส่วนใหญ่ "
                    "แต่ควรพิจารณา Transform ถ้า |Skew| ≥ 1"
                )
            else:
                skewness_insight += (
                    "\n\n**เกณฑ์การตัดสิน** (อ้างอิง Topic 2): "
                    "|Skew| < 0.5 = ใกล้ Normal — เหมาะสำหรับทุกโมเดล ไม่จำเป็นต้อง Transform"
                )

            st.info(f"**{selected_distribution_column}:** {skewness_insight}")

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
                color_discrete_sequence=["#0082CE"],
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
            color_discrete_sequence=["#0082CE"],
        )
        bar_chart_figure.update_layout(template="plotly_dark", height=450, showlegend=False)
        st.plotly_chart(bar_chart_figure, width="stretch")

        # Insight สำหรับ Categorical
        number_of_unique_categories = dataframe[selected_distribution_column].nunique()
        top_category_value = category_counts.iloc[0][selected_distribution_column] if len(category_counts) > 0 else "N/A"
        top_category_count = int(category_counts.iloc[0]["count"]) if len(category_counts) > 0 else 0
        top_category_percentage = (top_category_count / len(dataframe) * 100) if len(dataframe) > 0 else 0

        categorical_insight = f"มี **{number_of_unique_categories}** ค่า Unique — ค่าที่พบมากสุดคือ **{top_category_value}** ({top_category_percentage:.1f}%)"

        # ตรวจ Class Imbalance (ถ้าเป็น Target)
        if selected_distribution_column == target_column and number_of_unique_categories <= 20:
            minimum_class_count = int(category_counts["count"].min())
            maximum_class_count = int(category_counts["count"].max())
            if maximum_class_count > 3 * minimum_class_count:
                categorical_insight += (
                    "\n\n**Class Imbalance:** ค่าที่พบมากสุดมากกว่าค่าที่พบน้อยสุด "
                    f"ถึง {maximum_class_count / minimum_class_count:.1f} เท่า — อาจต้องจัดการก่อนสร้างโมเดล"
                )

        st.info(f"**{selected_distribution_column}:** {categorical_insight}")
