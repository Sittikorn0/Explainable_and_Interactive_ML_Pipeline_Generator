# Libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew

# Logic Import
from backend.function.data_type.dtype_detection import actual_type, ml_category
from backend.function.analyzer.task_detection import detect_task


def render_eda_target_info(dataframe: pd.DataFrame, target_column: str):
    """แสดงข้อมูล Target Column ในหน้า EDA แบบ Minimalist"""
    dtype = str(actual_type(dataframe[target_column]))
    unique_count = dataframe[target_column].nunique()
    task_type = detect_task(dataframe, target_column)
    
    # Minimal badge style
    badge = lambda text, color: f'<span style="background:rgba({color}, 0.1); color:rgb({color}); padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-left:8px; border:1px solid rgba({color}, 0.2); text-transform:uppercase;">{text}</span>'
    
    st.markdown(f"""
        <div style="background: rgba(122, 162, 247, 0.05); border: 1px solid rgba(122, 162, 247, 0.12); 
        border-radius: 10px; padding: 20px 28px; margin: 15px 0 30px 0; display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 16px; line-height: 1;">
                <span style="color: #7AA2F7; font-size: 0.95rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase;">Target:</span>
                <span style="font-family: 'JetBrains Mono', 'Roboto Mono', monospace; font-size: 1.25rem; color: #f8fafc; font-weight: 700;">{target_column}</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px; line-height: 1;">
                {badge(dtype, "122, 162, 247")}
                {badge(f"Unique: {unique_count}", "187, 154, 247")}
                {badge(task_type, "158, 206, 106")}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
def render_eda_profile_tab(dataframe: pd.DataFrame, target_column: str, outlier_details: list):
    st.subheader("Data Profile")

    outlier_dictionary = {item["Column"]: item for item in outlier_details}
    profile_data_list = []

    for column in dataframe.columns:
        data_series = dataframe[column]
        actual_data_type = actual_type(data_series)
        is_target_column = column == target_column
        profile_data_list.append({
            "Column": column,
            "Data Types": actual_data_type,
            "ML Category": ml_category(actual_data_type, is_target_column),
            "Missing": int(data_series.isnull().sum()),
            "Outliers": outlier_dictionary.get(column, {}).get("Outliers", 0),
            "Unique": int(data_series.nunique()),
        })

    st.dataframe(
        pd.DataFrame(profile_data_list),
        width="stretch",
        hide_index=True,
        column_config={
            "Column": st.column_config.TextColumn("Column"),
            "Data Types": st.column_config.TextColumn("Data Types", width="small"),
            "ML Category": st.column_config.TextColumn(
                "ML Category", width="medium"
            ),
            "Missing": st.column_config.NumberColumn("Missing", format="%d"),
            "Unique": st.column_config.ProgressColumn(
                "Unique", min_value=0, max_value=dataframe.shape[0], format="%d"
            ),
            "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
        },
    )

    with st.expander("Attribute Types & ML Category คืออะไร?"):
        st.markdown(
            "**ประเภทข้อมูลตาม Machine Learning**:\n\n"
            "| ประเภท | ความหมาย | ตัวอย่าง | ค่ากลางที่ใช้ได้ |\n"
            "|--------|---------|---------|----------------|\n"
            "| **Categorical/Nominal** | ข้อมูลเชิงหมวดหมู่ ไม่มีลำดับ | สีผม, เพศ, จังหวัด | Mode เท่านั้น |\n"
            "| **Numeric/Discrete** | ตัวเลขจำนวนเต็ม นับได้ | จำนวนสินค้า, อายุ | Mean, Median, Mode |\n"
            "| **Numeric/Continuous** | ตัวเลขทศนิยม วัดได้ | น้ำหนัก, อุณหภูมิ | Mean, Median |\n"
            "| **Datetime** | วันที่/เวลา | 2024-01-01 |  |\n\n"
            "**Target** คือคอลัมน์ที่ต้องการทำนาย ซึ่งเลือกไว้ในขั้นตอน Upload"
        )

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

def render_eda_distributions_tab(dataframe: pd.DataFrame, target_column: str):
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
                f" มีข้อมูลทั้งหมด **{len(datetime_series):,}** แถว"
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
        
MODERN_PALETTE = ["#7AA2F7", "#BB9AF7", "#7DCFFF", "#F7768E", "#9ECE6A", "#E0AF68", "#FF9E64", "#2AC3DE"]

def render_relationships_tab(dataframe: pd.DataFrame, target_column: str):
    st.subheader("Relationships & Redundancy")

    # Feature vs Target
    st.write("**Feature vs Target**")
    feature_columns_list = [column for column in dataframe.columns if column != target_column]
    is_target_numeric = pd.api.types.is_numeric_dtype(dataframe[target_column])
    is_target_categorical = not is_target_numeric

    if feature_columns_list:
        selected_feature = st.selectbox(
            "เลือก Feature",
            feature_columns_list,
            key="eda_feature_vs_target",
            label_visibility="collapsed",
        )
        is_feature_numeric = pd.api.types.is_numeric_dtype(dataframe[selected_feature])
        is_feature_datetime = (
            pd.api.types.is_datetime64_any_dtype(dataframe[selected_feature])
            or (dataframe[selected_feature].dtype == object and actual_type(dataframe[selected_feature]) == "datetime")
        )

        if is_feature_datetime:
            relationship_datetime_granularity = st.radio(
                "Granularity",
                ["Year", "Month", "Day"],
                horizontal=True,
                key="eda_rel_dt_granularity",
            )
            datetime_subset = dataframe[[selected_feature, target_column]].dropna().copy()
            if not pd.api.types.is_datetime64_any_dtype(datetime_subset[selected_feature]):
                datetime_subset[selected_feature] = pd.to_datetime(
                    datetime_subset[selected_feature], format="mixed", dayfirst=False, errors="coerce"
                )
            datetime_subset = datetime_subset.dropna(subset=[selected_feature])
            if relationship_datetime_granularity == "Year":
                datetime_subset["_period"] = datetime_subset[selected_feature].dt.year.astype(str)
            elif relationship_datetime_granularity == "Month":
                datetime_subset["_period"] = datetime_subset[selected_feature].dt.to_period("M").astype(str)
            else:
                datetime_subset["_period"] = datetime_subset[selected_feature].dt.to_period("D").astype(str)

            if is_target_numeric:
                period_means_dataframe = datetime_subset.groupby("_period")[target_column].mean().reset_index()
                period_means_dataframe.columns = [selected_feature, target_column]
                relationship_line_figure = px.line(
                    period_means_dataframe,
                    x=selected_feature,
                    y=target_column,
                    markers=True,
                    color_discrete_sequence=["#7AA2F7"],
                )
                relationship_line_figure.update_layout(
                    template="plotly_dark",
                    height=380,
                    yaxis_title=f"Mean {target_column}",
                )
                st.plotly_chart(relationship_line_figure, width="stretch")
                st.caption(f"ค่าเฉลี่ยของ {target_column} แต่ละ {relationship_datetime_granularity}")
            else:
                period_class_counts_dataframe = datetime_subset.groupby(["_period", target_column]).size().reset_index(name="count")
                period_class_counts_dataframe.columns = [selected_feature, target_column, "count"]
                relationship_line_figure = px.line(
                    period_class_counts_dataframe,
                    x=selected_feature,
                    y="count",
                    color=target_column,
                    markers=True,
                    color_discrete_sequence=MODERN_PALETTE,
                )
                relationship_line_figure.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(relationship_line_figure, width="stretch")
                st.caption(f"จำนวนแถวแต่ละ {target_column} class ตาม {relationship_datetime_granularity}")

        elif is_feature_numeric and is_target_numeric:
            relationship_scatter_figure = px.scatter(
                dataframe,
                x=selected_feature,
                y=target_column,
                opacity=0.6,
                color_discrete_sequence=["#7AA2F7"],
                trendline="ols",
                trendline_color_override="#F7768E",
            )
            relationship_scatter_figure.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(relationship_scatter_figure, width="stretch")
        elif is_feature_numeric and is_target_categorical:
            median_ordering = (
                dataframe.groupby(target_column)[selected_feature]
                .median()
                .sort_values()
                .index.tolist()
            )
            relationship_box_figure = px.box(
                dataframe,
                x=target_column,
                y=selected_feature,
                color=target_column,
                color_discrete_sequence=MODERN_PALETTE,
                category_orders={target_column: median_ordering},
            )
            relationship_box_figure.update_layout(template="plotly_dark", height=380, showlegend=False)
            st.plotly_chart(relationship_box_figure, width="stretch")
        elif not is_feature_numeric and is_target_numeric:
            number_of_feature_categories = dataframe[selected_feature].nunique()
            plotting_dataframe = dataframe if number_of_feature_categories <= 20 else dataframe[dataframe[selected_feature].isin(
                dataframe[selected_feature].value_counts().head(20).index
            )]
            median_ordering = (
                plotting_dataframe.groupby(selected_feature)[target_column]
                .median()
                .sort_values()
                .index.tolist()
            )
            relationship_box_figure = px.box(
                plotting_dataframe,
                x=selected_feature,
                y=target_column,
                color=selected_feature,
                color_discrete_sequence=MODERN_PALETTE,
                category_orders={selected_feature: median_ordering},
            )
            relationship_box_figure.update_layout(template="plotly_dark", height=380, showlegend=False)
            if number_of_feature_categories > 20:
                st.caption("แสดงแค่ 20 categories ที่พบมากสุด")
            st.plotly_chart(relationship_box_figure, width="stretch")
        else:
            number_of_feature_categories = dataframe[selected_feature].nunique()
            plotting_dataframe = dataframe if number_of_feature_categories <= 15 else dataframe[dataframe[selected_feature].isin(
                dataframe[selected_feature].value_counts().head(15).index
            )]
            category_ordering = plotting_dataframe[selected_feature].value_counts().index.tolist()
            cross_tabulation = plotting_dataframe.groupby([selected_feature, target_column]).size().reset_index(name="count")
            relationship_bar_figure = px.bar(
                cross_tabulation,
                x=selected_feature,
                y="count",
                color=target_column,
                barmode="group",
                color_discrete_sequence=MODERN_PALETTE,
                category_orders={selected_feature: category_ordering},
            )
            relationship_bar_figure.update_layout(template="plotly_dark", height=380)
            if number_of_feature_categories > 15:
                st.caption("แสดงแค่ 15 categories ที่พบมากสุด")
            st.plotly_chart(relationship_bar_figure, width="stretch")
    else:
        st.info("ไม่มี Feature columns ให้แสดง")

    st.divider()

    correlation_column1, correlation_column2 = st.columns(2)

    with correlation_column1:
        st.write("**Feature-Target Correlation**")
        all_numeric_dataframe = dataframe.select_dtypes(include=[np.number])
        if is_target_numeric and target_column in all_numeric_dataframe.columns:
            feature_numeric_columns = [column for column in all_numeric_dataframe.columns if column != target_column]
            if feature_numeric_columns:
                correlation_with_target = (
                    all_numeric_dataframe[feature_numeric_columns]
                    .corrwith(dataframe[target_column])
                    .dropna()
                    .sort_values(key=abs, ascending=True)
                )
                target_correlation_dataframe = pd.DataFrame({
                    "Feature": correlation_with_target.index,
                    "Correlation": correlation_with_target.values,
                })
                target_correlation_dataframe["color"] = target_correlation_dataframe["Correlation"].apply(
                    lambda value: "positive" if value >= 0 else "negative"
                )
                target_correlation_bar_figure = px.bar(
                    target_correlation_dataframe,
                    x="Correlation",
                    y="Feature",
                    orientation="h",
                    color="color",
                    color_discrete_map={"positive": "#9ECE6A", "negative": "#F7768E"},
                    range_x=[-1, 1],
                    text=target_correlation_dataframe["Correlation"].round(2),
                )
                target_correlation_bar_figure.update_traces(textposition="outside")
                target_correlation_bar_figure.update_layout(
                    template="plotly_dark",
                    height=max(300, 35 * len(target_correlation_dataframe)),
                    showlegend=False,
                    xaxis_title="Pearson r",
                )
                st.plotly_chart(target_correlation_bar_figure, width="stretch")
                st.caption(
                    "แท่งสีเขียว = สัมพันธ์เชิงบวกกับ target  |  "
                    "แท่งสีแดง = สัมพันธ์เชิงลบ  |  "
                    "ยิ่งแท่งยาว ยิ่งสัมพันธ์กันมาก"
                )
            else:
                st.info("ไม่มี Numeric Feature columns")
        else:
            feature_numeric_columns = [column for column in all_numeric_dataframe.columns if column != target_column]
            if feature_numeric_columns and dataframe[target_column].nunique() <= 20:
                class_mean_values = dataframe.groupby(target_column)[feature_numeric_columns].mean()
                spread_ordering = (
                    (class_mean_values.max() - class_mean_values.min())
                    .sort_values(ascending=True)
                    .index.tolist()
                )
                mean_by_class_dataframe = (
                    class_mean_values
                    .T
                    .reset_index()
                    .rename(columns={"index": "Feature"})
                )
                melted_mean_dataframe = mean_by_class_dataframe.melt(id_vars="Feature", var_name="Class", value_name="Mean")
                class_mean_bar_figure = px.bar(
                    melted_mean_dataframe,
                    x="Mean",
                    y="Feature",
                    color="Class",
                    orientation="h",
                    barmode="group",
                    color_discrete_sequence=MODERN_PALETTE,
                    category_orders={"Feature": spread_ordering},
                )
                class_mean_bar_figure.update_layout(
                    template="plotly_dark",
                    height=max(300, 40 * len(feature_numeric_columns)),
                )
                st.plotly_chart(class_mean_bar_figure, width="stretch")
                st.caption("ค่าเฉลี่ยของแต่ละ Feature แยกตาม Target Class  ถ้า class ต่างกันมาก แปลว่า feature นั้นช่วยแยก class ได้ดี")
            else:
                st.info("ไม่สามารถแสดง Feature-Target Correlation สำหรับข้อมูลประเภทนี้ได้")

    with correlation_column2:
        st.write("**Correlation Heatmap**")
        all_numeric_dataframe = dataframe.select_dtypes(include=[np.number])
        if all_numeric_dataframe.shape[1] > 1:
            number_of_numeric_columns = all_numeric_dataframe.shape[1]
            heatmap_calculated_height = max(400, min(800, 30 * number_of_numeric_columns))
            show_heatmap_text = ".2f" if number_of_numeric_columns <= 20 else False
            correlation_matrix = all_numeric_dataframe.corr()
            heatmap_figure = px.imshow(
                correlation_matrix,
                text_auto=show_heatmap_text,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
            )
            heatmap_figure.update_layout(template="plotly_dark", height=heatmap_calculated_height)
            if number_of_numeric_columns > 15:
                st.caption(f"มี {number_of_numeric_columns} numeric columns ซ่อนตัวเลขในตาราง hover เพื่ออ่านค่า")
            st.plotly_chart(heatmap_figure, width="stretch")

            # Detect High Correlation
            highly_correlated_pairs = []
            for index_i in range(len(correlation_matrix.columns)):
                for index_j in range(index_i + 1, len(correlation_matrix.columns)):
                    correlation_value = correlation_matrix.iloc[index_i, index_j]
                    if abs(correlation_value) > 0.8:
                        highly_correlated_pairs.append(
                            (correlation_matrix.columns[index_i], correlation_matrix.columns[index_j], round(correlation_value, 2))
                        )

            if highly_correlated_pairs:
                correlated_pairs_bullets = "\n".join(
                    [f"- **{col_a}** และ **{col_b}** (ความสัมพันธ์: {corr_val})" for col_a, col_b, corr_val in highly_correlated_pairs]
                )
                st.warning(
                    "[!] **แจ้งเตือนปัญหา Multicollinearity (ข้อมูลซ้ำซ้อน)**\n\n"
                    "พบคอลัมน์ที่มีความสัมพันธ์กันเองสูงมาก (|r| > 0.8):\n"
                    f"{correlated_pairs_bullets}\n\n"
                    "**ผลกระทบ:** อาจทำให้โมเดลที่อ่อนไหว (เช่น Linear/Logistic Regression) สับสนและทำนายผิดพลาดได้\n\n"
                    "**คำแนะนำ:** ควรพิจารณา **ลบคอลัมน์ใดคอลัมน์หนึ่ง** ในคู่ที่ซ้ำซ้อนกันออก"
                )
        else:
            st.info("Need more numeric columns for correlation.")