import streamlit as st
import pandas as pd
from data_prepare.logic.data_type_detection import actual_type, ml_category

def render_profile_tab(dataframe: pd.DataFrame, target_column: str, outlier_details: list):
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
            "| **Datetime** | วันที่/เวลา | 2024-01-01 | — |\n\n"
            "**Target** คือคอลัมน์ที่ต้องการทำนาย ซึ่งเลือกไว้ในขั้นตอน Upload"
        )
