import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import skew
from data_prepare.features.data_distribute import data_distribution
from data_prepare.features.data_type_detection import actual_type, ml_category
from data_prepare.features.target_col import describe_target


def _skew_insight(col_skew: float) -> str:
    """สร้างข้อความอธิบาย skewness"""
    abs_skew = abs(col_skew)
    if abs_skew < 0.5:
        shape = "ใกล้ Normal (สมมาตร)"
    elif abs_skew < 1:
        shape = "เบ้เล็กน้อย (Moderately Skewed)"
    else:
        shape = "เบ้มาก (Highly Skewed)"

    direction = ""
    if col_skew > 0.5:
        direction = " → หางยาวไปทางขวา (Right-skewed)"
    elif col_skew < -0.5:
        direction = " → หางยาวไปทางซ้าย (Left-skewed)"

    return f"Skewness = {col_skew:.2f} → {shape}{direction}"

def _fmt_pct(count: int, pct: float) -> str:
    if count == 0:
        return "0 (0.0%)"
    pct_str = f"{pct:.1f}%" if pct >= 0.1 else "< 0.1%"
    return f"{count:,} ({pct_str})"


def render_eda():
    from app import page_header

    page_header("Exploratory Data Analysis", "สำรวจและทำความเข้าใจข้อมูลก่อนสร้างโมเดล")

    if st.session_state.get("main_df") is None:
        from app import navigate
        navigate("upload")
        return

    has_working_df = "working_df" in st.session_state
    df = st.session_state["working_df"] if has_working_df else st.session_state["main_df"]
    file_name = st.session_state.get("last_uploaded_file", "Unknown File")
    is_cleaned = has_working_df and st.session_state.get("cleaning_confirmed")

    st.info(f"**Current Dataset:** {file_name}")
    with st.expander("Cleaned Data" if is_cleaned else "Raw Data"):
        st.dataframe(df, width="stretch")

    # Dataset Overview
    st.subheader("Dataset Overview")
    # ใช้ขอบเขต Outlier เดียวกับหน้า Cleaning (fixed_bounds) ถ้ามี
    # เพื่อให้ตัวเลข Outlier card ตรงกันทั้งสองหน้า
    fixed_bounds = st.session_state.get("original_outlier_bounds", None)

    _bounds_sig = id(fixed_bounds) if fixed_bounds else 0
    _dist_key = ("_eda_dist_cache", df.shape, int(pd.util.hash_pandas_object(df).sum()), _bounds_sig)
    if st.session_state.get("_eda_dist_key") != _dist_key:
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(df, fixed_bounds=fixed_bounds)
        st.session_state["_eda_dist_key"] = _dist_key
        st.session_state["_eda_dist_result"] = (total_outl, outls_details)
    else:
        total_outl, outls_details = st.session_state["_eda_dist_result"]

    total_cells = df.size
    total_missing = df.isnull().sum().sum()
    missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0
    duplicate_count = int(df.duplicated().sum())
    dup_pct = (duplicate_count / df.shape[0] * 100) if df.shape[0] > 0 else 0
    outlier_pct = (total_outl / total_cells * 100) if total_cells > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing Values", _fmt_pct(total_missing, missing_pct))
    m4.metric("Duplicate Rows", _fmt_pct(duplicate_count, dup_pct))
    m5.metric("Outliers", _fmt_pct(total_outl, outlier_pct))

    # ดึง target column จากที่เลือกไว้ในหน้า Upload พร้อม fallback ถ้าไม่มีหรือไม่ valid
    target_col = st.session_state.get("target_col", df.columns[-1])
    if target_col not in df.columns:
        target_col = df.columns[-1]
        st.warning(f"Target column ที่เลือกไว้ไม่พบใน dataset — ใช้ **{target_col}** แทน")
    st.info(f"**Target Column:** {target_col}  \n{describe_target(df, target_col)}")

    # Proactive class imbalance check — แสดงในหน้า Overview โดยไม่ต้องรอให้ user เลือก column
    if not pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique() <= 20:
        target_counts = df[target_col].value_counts()
        min_count = int(target_counts.min())
        max_count = int(target_counts.max())
        if max_count > 3 * min_count:
            st.warning(
                f"**Class Imbalance ตรวจพบใน Target '{target_col}':** "
                f"ค่าที่พบมากสุดมากกว่าค่าที่พบน้อยสุดถึง **{max_count / min_count:.1f} เท่า** "
                "— อาจต้องจัดการก่อนสร้างโมเดล เช่น Oversampling, SMOTE, หรือปรับ class_weight"
            )

    tab1, tab2, tab3 = st.tabs(
        ["Profile", "Distributions", "Relationships"], width="stretch"
    )

    # Profile
    with tab1:
        st.subheader("Data Profile")

        outlier_dict = {item["Column"]: item for item in outls_details}
        profile_list = []

        for col in df.columns:
            series = df[col]
            actual = actual_type(series)
            is_target = col == target_col
            profile_list.append({
                "Column": col,
                "Data Types": actual,
                "ML Category": ml_category(actual, is_target),
                "Missing": int(series.isnull().sum()),
                "Outliers": outlier_dict.get(col, {}).get("Outliers", 0),
                "Unique": int(series.nunique()),
            })

        st.dataframe(
            pd.DataFrame(profile_list),
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
                    "Unique", min_value=0, max_value=df.shape[0], format="%d"
                ),
                "Outliers": st.column_config.NumberColumn("Outliers", format="%d"),
            },
        )

        # คำอธิบาย Attribute Types
        with st.expander("Attribute Types & ML Category คืออะไร? (อ้างอิง Topic 2)"):
            st.markdown(
                "**ประเภทข้อมูลตาม Machine Learning** (อ้างอิง Topic 2 — Getting to Know Your Data):\n\n"
                "| ประเภท | ความหมาย | ตัวอย่าง | ค่ากลางที่ใช้ได้ |\n"
                "|--------|---------|---------|----------------|\n"
                "| **Categorical/Nominal** | ข้อมูลเชิงหมวดหมู่ ไม่มีลำดับ | สีผม, เพศ, จังหวัด | Mode เท่านั้น |\n"
                "| **Numeric/Discrete** | ตัวเลขจำนวนเต็ม นับได้ | จำนวนสินค้า, อายุ | Mean, Median, Mode |\n"
                "| **Numeric/Continuous** | ตัวเลขทศนิยม วัดได้ | น้ำหนัก, อุณหภูมิ | Mean, Median |\n"
                "| **Datetime** | วันที่/เวลา | 2024-01-01 | — |\n\n"
                "**Target** คือคอลัมน์ที่ต้องการทำนาย ซึ่งเลือกไว้ในขั้นตอน Upload\n\n"
                "> Topic 2 ยังอธิบายถึง **Ordinal** (ข้อมูลที่มีลำดับ เช่น Low/Medium/High) "
                "และ **Binary** (ข้อมูล 2 ค่า เช่น Yes/No) ซึ่งระบบจัดอยู่ในกลุ่ม Categorical"
            )

    # Distributions
    with tab2:
        st.subheader("Data Distributions")

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        # รวม string columns ที่ actual_type ตรวจว่าเป็น datetime เพื่อให้สอดคล้องกับ Profile tab
        dt_cols = [
            c for c in df.columns
            if pd.api.types.is_datetime64_any_dtype(df[c])
            or (df[c].dtype == object and actual_type(df[c]) == "datetime")
        ]
        dt_col_set = set(dt_cols)
        cat_cols = [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if c not in dt_col_set
        ]
        dist_cols = num_cols + cat_cols + dt_cols

        if not dist_cols:
            st.info("ไม่มีคอลัมน์ประเภท Numeric, Categorical หรือ Datetime สำหรับแสดง Distribution")
        else:
            selected_col = st.selectbox(
                "Select column to visualize",
                dist_cols,
                key="eda_dist_col",
            )

            if selected_col in num_cols:
                fig = px.histogram(
                    df,
                    x=selected_col,
                    marginal="box",
                    nbins=30,
                    color_discrete_sequence=["#0082CE"],
                )
                fig.update_layout(template="plotly_dark", height=450, showlegend=False)
                st.plotly_chart(fig, width="stretch")

                # Insight สำหรับ Numeric
                col_data = df[selected_col].dropna()
                if len(col_data) < 2 or col_data.nunique() <= 1:
                    st.info(f"**{selected_col}:** ข้อมูลไม่เพียงพอสำหรับวิเคราะห์ Skewness")
                else:
                    col_skew = float(skew(col_data))
                    insight = _skew_insight(col_skew)

                    if abs(col_skew) >= 1:
                        col_min = float(col_data.min())
                        if col_min <= 0:
                            transform_rec = "**Yeo-Johnson Transformation** (รองรับค่า 0 และค่าลบ)"
                        else:
                            transform_rec = "Log, Box-Cox, หรือ Yeo-Johnson Transformation"
                        insight += (
                            f"\n\n**คำแนะนำ** (อ้างอิง Topic 9): ข้อมูลเบ้มาก (|Skew| ≥ 1) "
                            f"อาจต้อง Transform ก่อนสร้างโมเดล เช่น {transform_rec}"
                        )
                    elif abs(col_skew) >= 0.5:
                        insight += (
                            "\n\n**เกณฑ์การตัดสิน** (อ้างอิง Topic 2): "
                            "|Skew| ≥ 0.5 = เบ้เล็กน้อย — ยังใช้ได้กับโมเดลส่วนใหญ่ "
                            "แต่ควรพิจารณา Transform ถ้า |Skew| ≥ 1"
                        )
                    else:
                        insight += (
                            "\n\n**เกณฑ์การตัดสิน** (อ้างอิง Topic 2): "
                            "|Skew| < 0.5 = ใกล้ Normal — เหมาะสำหรับทุกโมเดล ไม่จำเป็นต้อง Transform"
                        )

                    st.info(f"**{selected_col}:** {insight}")

            elif selected_col in dt_cols:
                # ── Datetime Distribution ──────────────────────
                # parse string columns ที่ยังไม่เป็น datetime64 ให้เป็นก่อน
                if pd.api.types.is_datetime64_any_dtype(df[selected_col]):
                    dt_series = df[selected_col].dropna()
                else:
                    dt_series = pd.to_datetime(
                        df[selected_col], format="mixed", dayfirst=False, errors="coerce"
                    ).dropna()
                if len(dt_series) == 0:
                    st.info(f"**{selected_col}:** ไม่มีข้อมูล valid (ทั้งหมดเป็น NaN)")
                else:
                    granularity = st.radio(
                        "Granularity",
                        ["Year", "Month", "Day"],
                        horizontal=True,
                        key="eda_dt_granularity",
                    )
                    if granularity == "Year":
                        period_counts = dt_series.dt.year.value_counts().sort_index().reset_index()
                        period_counts.columns = [selected_col, "count"]
                    elif granularity == "Month":
                        period_counts = (
                            dt_series.dt.to_period("M")
                            .astype(str)
                            .value_counts()
                            .sort_index()
                            .reset_index()
                        )
                        period_counts.columns = [selected_col, "count"]
                    else:
                        period_counts = (
                            dt_series.dt.to_period("D")
                            .astype(str)
                            .value_counts()
                            .sort_index()
                            .reset_index()
                        )
                        period_counts.columns = [selected_col, "count"]

                    fig = px.line(
                        period_counts,
                        x=selected_col,
                        y="count",
                        markers=True,
                        color_discrete_sequence=["#0082CE"],
                    )
                    fig.update_layout(template="plotly_dark", height=450)
                    st.plotly_chart(fig, width="stretch")

                    date_min = dt_series.min().strftime("%Y-%m-%d")
                    date_max = dt_series.max().strftime("%Y-%m-%d")
                    st.info(
                        f"**{selected_col}:** ช่วงข้อมูลตั้งแต่ **{date_min}** ถึง **{date_max}** "
                        f"— มีข้อมูลทั้งหมด **{len(dt_series):,}** แถว"
                    )

            else:
                counts = df[selected_col].value_counts().reset_index()
                counts.columns = [selected_col, "count"]
                fig = px.bar(
                    counts.head(20),
                    x=selected_col,
                    y="count",
                    color_discrete_sequence=["#0082CE"],
                )
                fig.update_layout(template="plotly_dark", height=450, showlegend=False)
                st.plotly_chart(fig, width="stretch")

                # Insight สำหรับ Categorical
                n_unique = df[selected_col].nunique()
                top_value = counts.iloc[0][selected_col] if len(counts) > 0 else "N/A"
                top_count = int(counts.iloc[0]["count"]) if len(counts) > 0 else 0
                top_pct = (top_count / len(df) * 100) if len(df) > 0 else 0

                insight = f"มี **{n_unique}** ค่า Unique — ค่าที่พบมากสุดคือ **{top_value}** ({top_pct:.1f}%)"

                # ตรวจ Class Imbalance (ถ้าเป็น Target)
                if selected_col == target_col and n_unique <= 20:
                    min_count = int(counts["count"].min())
                    max_count = int(counts["count"].max())
                    if max_count > 3 * min_count:
                        insight += (
                            "\n\n**Class Imbalance:** ค่าที่พบมากสุดมากกว่าค่าที่พบน้อยสุด "
                            f"ถึง {max_count / min_count:.1f} เท่า — อาจต้องจัดการก่อนสร้างโมเดล"
                        )

                st.info(f"**{selected_col}:** {insight}")

    # Relationships
    with tab3:
        st.subheader("Relationships & Redundancy")

        # ── Feature vs Target ─────────────────────────────────
        st.write("**Feature vs Target**")
        feature_cols = [c for c in df.columns if c != target_col]
        target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
        target_is_cat = not target_is_numeric

        if feature_cols:
            sel_feature = st.selectbox(
                "เลือก Feature",
                feature_cols,
                key="eda_feature_vs_target",
                label_visibility="collapsed",
            )
            feature_is_numeric = pd.api.types.is_numeric_dtype(df[sel_feature])
            feature_is_datetime = (
                pd.api.types.is_datetime64_any_dtype(df[sel_feature])
                or (df[sel_feature].dtype == object and actual_type(df[sel_feature]) == "datetime")
            )

            if feature_is_datetime:
                # Datetime Feature vs Numeric/Categorical Target → Line chart over time
                granularity = st.radio(
                    "Granularity",
                    ["Year", "Month", "Day"],
                    horizontal=True,
                    key="eda_rel_dt_granularity",
                )
                dt_subset = df[[sel_feature, target_col]].dropna().copy()
                # parse string datetime ให้เป็น datetime64 ก่อนใช้ .dt accessor
                if not pd.api.types.is_datetime64_any_dtype(dt_subset[sel_feature]):
                    dt_subset[sel_feature] = pd.to_datetime(
                        dt_subset[sel_feature], format="mixed", dayfirst=False, errors="coerce"
                    )
                dt_subset = dt_subset.dropna(subset=[sel_feature])
                if granularity == "Year":
                    dt_subset["_period"] = dt_subset[sel_feature].dt.year.astype(str)
                elif granularity == "Month":
                    dt_subset["_period"] = dt_subset[sel_feature].dt.to_period("M").astype(str)
                else:
                    dt_subset["_period"] = dt_subset[sel_feature].dt.to_period("D").astype(str)

                if target_is_numeric:
                    period_means = dt_subset.groupby("_period")[target_col].mean().reset_index()
                    period_means.columns = [sel_feature, target_col]
                    fig_ft = px.line(
                        period_means,
                        x=sel_feature,
                        y=target_col,
                        markers=True,
                        color_discrete_sequence=["#0082CE"],
                    )
                    fig_ft.update_layout(
                        template="plotly_dark",
                        height=380,
                        yaxis_title=f"Mean {target_col}",
                    )
                    st.plotly_chart(fig_ft, width="stretch")
                    st.caption(f"ค่าเฉลี่ยของ {target_col} แต่ละ {granularity}")
                else:
                    period_class_counts = dt_subset.groupby(["_period", target_col]).size().reset_index(name="count")
                    period_class_counts.columns = [sel_feature, target_col, "count"]
                    fig_ft = px.line(
                        period_class_counts,
                        x=sel_feature,
                        y="count",
                        color=target_col,
                        markers=True,
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig_ft.update_layout(template="plotly_dark", height=380)
                    st.plotly_chart(fig_ft, width="stretch")
                    st.caption(f"จำนวนแถวแต่ละ {target_col} class ตาม {granularity}")

            elif feature_is_numeric and target_is_numeric:
                # Numeric vs Numeric → Scatter + OLS trendline
                fig_ft = px.scatter(
                    df,
                    x=sel_feature,
                    y=target_col,
                    opacity=0.6,
                    color_discrete_sequence=["#0082CE"],
                    trendline="ols",
                    trendline_color_override="#f87171",
                )
                fig_ft.update_layout(template="plotly_dark", height=380)
                st.plotly_chart(fig_ft, width="stretch")
            elif feature_is_numeric and target_is_cat:
                # Numeric Feature vs Categorical Target → Box plot
                median_order = (
                    df.groupby(target_col)[sel_feature]
                    .median()
                    .sort_values()
                    .index.tolist()
                )
                fig_ft = px.box(
                    df,
                    x=target_col,
                    y=sel_feature,
                    color=target_col,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    category_orders={target_col: median_order},
                )
                fig_ft.update_layout(template="plotly_dark", height=380, showlegend=False)
                st.plotly_chart(fig_ft, width="stretch")
            elif not feature_is_numeric and target_is_numeric:
                # Categorical Feature vs Numeric Target → Box plot
                n_cats = df[sel_feature].nunique()
                plot_df = df if n_cats <= 20 else df[df[sel_feature].isin(
                    df[sel_feature].value_counts().head(20).index
                )]
                median_order = (
                    plot_df.groupby(sel_feature)[target_col]
                    .median()
                    .sort_values()
                    .index.tolist()
                )
                fig_ft = px.box(
                    plot_df,
                    x=sel_feature,
                    y=target_col,
                    color=sel_feature,
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    category_orders={sel_feature: median_order},
                )
                fig_ft.update_layout(template="plotly_dark", height=380, showlegend=False)
                if n_cats > 20:
                    st.caption("แสดงแค่ 20 categories ที่พบมากสุด")
                st.plotly_chart(fig_ft, width="stretch")
            else:
                # Categorical vs Categorical → Grouped bar
                n_cats = df[sel_feature].nunique()
                plot_df = df if n_cats <= 15 else df[df[sel_feature].isin(
                    df[sel_feature].value_counts().head(15).index
                )]
                cat_order = plot_df[sel_feature].value_counts().index.tolist()
                ct = plot_df.groupby([sel_feature, target_col]).size().reset_index(name="count")
                fig_ft = px.bar(
                    ct,
                    x=sel_feature,
                    y="count",
                    color=target_col,
                    barmode="group",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    category_orders={sel_feature: cat_order},
                )
                fig_ft.update_layout(template="plotly_dark", height=380)
                if n_cats > 15:
                    st.caption("แสดงแค่ 15 categories ที่พบมากสุด")
                st.plotly_chart(fig_ft, width="stretch")
        else:
            st.info("ไม่มี Feature columns ให้แสดง")

        st.divider()

        p_col1, p_col2 = st.columns(2)

        with p_col1:
            st.write("**Feature-Target Correlation**")
            numeric_df_all = df.select_dtypes(include=[np.number])
            if target_is_numeric and target_col in numeric_df_all.columns:
                feature_num_cols = [c for c in numeric_df_all.columns if c != target_col]
                if feature_num_cols:
                    corr_with_target = (
                        numeric_df_all[feature_num_cols]
                        .corrwith(df[target_col])
                        .dropna()
                        .sort_values(key=abs, ascending=True)
                    )
                    corr_df = pd.DataFrame({
                        "Feature": corr_with_target.index,
                        "Correlation": corr_with_target.values,
                    })
                    corr_df["color"] = corr_df["Correlation"].apply(
                        lambda x: "positive" if x >= 0 else "negative"
                    )
                    fig_corr_target = px.bar(
                        corr_df,
                        x="Correlation",
                        y="Feature",
                        orientation="h",
                        color="color",
                        color_discrete_map={"positive": "#4ade80", "negative": "#f87171"},
                        range_x=[-1, 1],
                        text=corr_df["Correlation"].round(2),
                    )
                    fig_corr_target.update_traces(textposition="outside")
                    fig_corr_target.update_layout(
                        template="plotly_dark",
                        height=max(300, 35 * len(corr_df)),
                        showlegend=False,
                        xaxis_title="Pearson r",
                    )
                    st.plotly_chart(fig_corr_target, width="stretch")
                    st.caption(
                        "แท่งสีเขียว = สัมพันธ์เชิงบวกกับ target  |  "
                        "แท่งสีแดง = สัมพันธ์เชิงลบ  |  "
                        "ยิ่งแท่งยาว ยิ่งสัมพันธ์กันมาก"
                    )
                else:
                    st.info("ไม่มี Numeric Feature columns")
            else:
                # Categorical target → แสดง mean ของ numeric features แยกตาม class
                feature_num_cols = [c for c in numeric_df_all.columns if c != target_col]
                if feature_num_cols and df[target_col].nunique() <= 20:
                    class_means = df.groupby(target_col)[feature_num_cols].mean()
                    # เรียง feature ตาม spread (max-min) ระหว่าง class → feature ที่แยก class ได้ดีอยู่บนสุด
                    spread_order = (
                        (class_means.max() - class_means.min())
                        .sort_values(ascending=True)
                        .index.tolist()
                    )
                    mean_by_class = (
                        class_means
                        .T
                        .reset_index()
                        .rename(columns={"index": "Feature"})
                    )
                    mean_melted = mean_by_class.melt(id_vars="Feature", var_name="Class", value_name="Mean")
                    fig_mean = px.bar(
                        mean_melted,
                        x="Mean",
                        y="Feature",
                        color="Class",
                        orientation="h",
                        barmode="group",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                        category_orders={"Feature": spread_order},
                    )
                    fig_mean.update_layout(
                        template="plotly_dark",
                        height=max(300, 40 * len(feature_num_cols)),
                    )
                    st.plotly_chart(fig_mean, width="stretch")
                    st.caption("ค่าเฉลี่ยของแต่ละ Feature แยกตาม Target Class — ถ้า class ต่างกันมาก แปลว่า feature นั้นช่วยแยก class ได้ดี")
                else:
                    st.info("ไม่สามารถแสดง Feature-Target Correlation สำหรับข้อมูลประเภทนี้ได้")

        with p_col2:
            st.write("**Correlation Heatmap**")
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                n_numeric_cols = numeric_df.shape[1]
                heatmap_height = max(400, min(800, 30 * n_numeric_cols))
                # ซ่อนตัวเลขเมื่อ column มากเกิน 15 เพราะ cell เล็กเกินอ่าน
                show_text = ".2f" if n_numeric_cols <= 15 else False
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr,
                    text_auto=show_text,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    range_color=[-1, 1],
                )
                fig_corr.update_layout(template="plotly_dark", height=heatmap_height)
                if n_numeric_cols > 15:
                    st.caption(f"มี {n_numeric_cols} numeric columns — ซ่อนตัวเลขในตาราง hover เพื่ออ่านค่า")
                st.plotly_chart(fig_corr, width="stretch")

                # Detect High Correlation
                high_corr_pairs = []
                for i in range(len(corr.columns)):
                    for j in range(i + 1, len(corr.columns)):
                        r = corr.iloc[i, j]
                        if abs(r) > 0.8:
                            high_corr_pairs.append(
                                (corr.columns[i], corr.columns[j], round(r, 2))
                            )

                if high_corr_pairs:
                    pairs_text = ", ".join(
                        [f"**{a}** ↔ **{b}** (r={r})" for a, b, r in high_corr_pairs]
                    )
                    st.warning(
                        f"**Multicollinearity ที่อาจเป็นปัญหา** (อ้างอิง Topic 2): {pairs_text}\n\n"
                        "คอลัมน์ที่มี Correlation สูง (|r| > 0.8) อาจให้ข้อมูลซ้ำซ้อนกัน "
                        "ซึ่งอาจทำให้โมเดลบางชนิด (เช่น Linear Regression) ทำงานได้ไม่ดี "
                        "อาจพิจารณาลบคอลัมน์ใดคอลัมน์หนึ่งออก"
                    )
            else:
                st.info("Need more numeric columns for correlation.")

    # Page Navigation
    col1, _, col2 = st.columns([0.8, 8, 0.8])
    with col1:
        if st.button("Back", type="secondary", width="stretch", key="back"):
            from app import navigate
            navigate("cleaning")
    with col2:
        if st.button("Next Step", type="primary", width="stretch"):
            from app import navigate
            navigate("transformation")