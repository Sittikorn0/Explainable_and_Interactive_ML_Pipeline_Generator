import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from features.data_distribute import data_distribution

def render_eda():
    from app import page_header
    page_header("Exploratory Data Analysis", "ตรวจสอบและสำรวจข้อมูลเบื้องต้น")
    
    if st.session_state.get('main_df') is not None:
        df = st.session_state['main_df']
        file_name = st.session_state.get('last_uploaded_file', 'Unknown File')
        
        st.info(f"**Current Dataset:** {file_name}")
        with st.expander("Raw Data"):
            st.dataframe(df, width="stretch")
        
        st.subheader("Dataset Overview")
        with st.spinner("Calculating Data..."):
            total_outl, outls_details = data_distribution(df)
        
        m_row1, m_row2, m_row3, m_row4, m_row5 = st.columns(5)
        m_row1.metric("Rows", f"{df.shape[0]:,}")
        m_row2.metric("Columns", df.shape[1])
        
        total_cells = df.size
        total_missing = df.isnull().sum().sum()
        missing_pct = (total_missing / total_cells) * 100 if total_cells > 0 else 0
        m_row3.metric(label="Missing Values", value=f"{total_missing:,} ({missing_pct:.1f}%)")
        
        total_rows = df.shape[0]
        duplicate_count = df.duplicated().sum()
        dup_pct = (duplicate_count / total_rows) * 100 if total_rows > 0 else 0
        m_row4.metric(label="Duplicate Rows", value=f"{duplicate_count:,} ({dup_pct:.1f}%)")
        
        outlier_pct = (total_outl / total_cells) * 100 if total_cells > 0 else 0
        m_row5.metric(label="Outliers", value=f"{total_outl:,} ({outlier_pct:.1f}%)")

        tab1, tab2, tab3 = st.tabs(["Profile", "Distributions", "Relationships"], width="stretch")
        
        with tab1:
            st.subheader("Data Profile")

            # --- Helpers (นิยามนอก loop เพื่อไม่สร้าง function ซ้ำทุก iteration) ---

            def _is_pseudo_int(s: pd.Series) -> bool:
                # float64 ที่ pandas บังคับเพราะมี NaN แต่ค่าจริงเป็นจำนวนเต็มทั้งหมด
                non_null = s.dropna()
                if len(non_null) == 0:
                    return False
                return bool((non_null % 1 == 0).all())

            def _actual_type(series: pd.Series) -> str:
                # คืน type ที่สื่อความหมายได้จริง ไม่ใช่ pandas internal dtype
                dtype = str(series.dtype)
                if dtype.startswith('int'):
                    return 'int'
                if dtype.startswith('float'):
                    return 'int' if _is_pseudo_int(series) else 'float'
                if dtype == 'bool':
                    return 'bool'
                if dtype == 'object':
                    try:
                        pd.to_datetime(series.dropna().head(20), format='mixed', dayfirst=False)
                        return 'datetime'
                    except (ValueError, TypeError):
                        pass
                    return 'string'
                return dtype

            def _ml_category(series: pd.Series, actual: str, is_target: bool) -> str:
                is_num = actual in ('int', 'float')
                if actual == 'datetime':
                    cat = 'Datetime'
                elif is_num:
                    cat = 'Numeric/Discrete' if actual == 'int' else 'Numeric/Continuous'
                else:
                    cat = 'Categorical/Nominal'
                return f"{cat} (Target)" if is_target else cat

            # --- Build profile table ---
            outlier_dict = {item['Column']: item for item in outls_details}
            target_col   = df.columns[-1]
            profile_list = []

            for col in df.columns:
                series    = df[col]
                actual    = _actual_type(series)
                is_target = (col == target_col)
                profile_list.append({
                    "Column"     : col,
                    "Data Types"       : actual,
                    "ML Category": _ml_category(series, actual, is_target),
                    "Missing"    : int(series.isnull().sum()),
                    "Outliers"   : outlier_dict.get(col, {}).get('Outliers', 0),
                    "Unique"     : int(series.nunique()),
                })

            profile_df = pd.DataFrame(profile_list)

            st.dataframe(
                profile_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Column"     : st.column_config.TextColumn("Column"),
                    "Data Types"       : st.column_config.TextColumn("Data Types", width="small"),
                    "ML Category": st.column_config.TextColumn("ML Category", width="medium"),
                    "Missing"    : st.column_config.NumberColumn("Missing", format="%d"),
                    "Unique"     : st.column_config.ProgressColumn(
                                        "Unique", min_value=0,
                                        max_value=df.shape[0], format="%d"),
                    "Outliers"   : st.column_config.NumberColumn("Outliers", format="%d"),
                }
            )
            
        with tab2:
            st.subheader("Data Distributions")
            
            # แยกประเภท Column
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            selected_col = st.selectbox("Select column to visualize", num_cols + cat_cols)
            
            if selected_col in num_cols:
                # Numeric: Histogram + Boxplot (ที่คุณทำไว้ดีอยู่แล้ว)
                fig = px.histogram(df, x=selected_col, marginal="box", 
                                nbins=30, color_discrete_sequence=["#0082CE"]) # ใช้สีขาวตามสไตล์ปุ่ม Primary
            else:
                # Categorical: Bar Chart (นับจำนวน Unique values)
                counts = df[selected_col].value_counts().reset_index()
                counts.columns = [selected_col, 'count']
                # แสดงแค่ Top 20 ถ้า Unique เยอะเกินไปเพื่อไม่ให้รก
                fig = px.bar(counts.head(20), x=selected_col, y='count', 
                            color_discrete_sequence=['#0082CE'])
            
            fig.update_layout(template="plotly_dark", height=450, showlegend=False)
            st.plotly_chart(fig, width="stretch")

        with tab3:
            st.subheader("Relationships & Redundancy")
            import plotly.graph_objects as go
            
            p_col1, p_col2 = st.columns(2)
            
            with p_col1:
                st.write("**Missing Value Scan**")
                # สร้าง Heatmap ของค่าว่าง
                # 1 = Missing, 0 = Not Missing
                missing_data = df.isnull().astype(int)
                fig_miss = px.imshow(
                    missing_data, 
                    labels=dict(x="Columns", y="Rows", color="Is Missing?"),
                    color_continuous_scale="Viridis"
                )
                fig_miss.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig_miss, width="stretch")

            with p_col2:
                st.write("**Correlation Heatmap**")
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] > 1:
                    corr = numeric_df.corr()
                    # ใช้ Heatmap แบบ Interactive
                    fig_corr = px.imshow(
                        corr,
                        text_auto=".2f", # แสดงตัวเลขในช่อง (ถ้าช่องไม่เล็กเกินไป)
                        aspect="auto",
                        color_continuous_scale="RdBu_r", # แดง-ขาว-น้ำเงิน
                        range_color=[-1, 1]
                    )
                    fig_corr.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_corr, width="stretch")
                else:
                    st.info("Need more numeric columns for correlation.")

        col1, space , col2 = st.columns([0.8, 8, 0.8])
        with col1:
            if st.button("Back",type="secondary", width="stretch", key="back"):
                st.query_params["step"] = "cleaning"
                st.rerun()
        with col2:
            if st.button("Next Step", type="primary", width="stretch"):
                st.query_params["step"] = "next"
                st.rerun()
            
    else:
        st.query_params["step"] = "upload"
        st.rerun()