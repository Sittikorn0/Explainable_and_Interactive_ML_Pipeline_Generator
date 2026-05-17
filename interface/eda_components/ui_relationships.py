import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from data_prepare.logic.data_type_detection import actual_type

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
                st.caption("ค่าเฉลี่ยของแต่ละ Feature แยกตาม Target Class — ถ้า class ต่างกันมาก แปลว่า feature นั้นช่วยแยก class ได้ดี")
            else:
                st.info("ไม่สามารถแสดง Feature-Target Correlation สำหรับข้อมูลประเภทนี้ได้")

    with correlation_column2:
        st.write("**Correlation Heatmap**")
        all_numeric_dataframe = dataframe.select_dtypes(include=[np.number])
        if all_numeric_dataframe.shape[1] > 1:
            number_of_numeric_columns = all_numeric_dataframe.shape[1]
            heatmap_calculated_height = max(400, min(800, 30 * number_of_numeric_columns))
            show_heatmap_text = ".2f" if number_of_numeric_columns <= 15 else False
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
                st.caption(f"มี {number_of_numeric_columns} numeric columns — ซ่อนตัวเลขในตาราง hover เพื่ออ่านค่า")
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
