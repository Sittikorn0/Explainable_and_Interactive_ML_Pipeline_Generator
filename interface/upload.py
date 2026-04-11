import streamlit as st
from features.loading_data import process_data, save_to_local

def render_upload():
    from app import page_header

    page_header(
        "Upload Dataset",
        "Support CSV, Excel, JSON ( Maximum size 200 MB )",
    )

    uploaded_file = st.file_uploader(
        "Upload",
        type=["csv", "xlsx", "xls", "json"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        # โหลดและ cache เฉพาะเมื่อเป็นไฟล์ใหม่
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state["last_uploaded_file"] != uploaded_file.name
        ):
            df = process_data(uploaded_file)
            if df is not None:
                st.session_state["main_df"] = df
                st.session_state["last_uploaded_file"] = uploaded_file.name
                save_to_local(df, uploaded_file.name)
            else:
                st.error("Failed to load data. Please check the file format and content.")
                return

        df = st.session_state.get("main_df")

        if df is not None:
            st.success(f"Load data from '{uploaded_file.name}' successfully!")

            col1, col2 = st.columns(2)
            col1.metric("Rows", f"{df.shape[0]:,}")
            col2.metric("Columns", f"{df.shape[1]}")

            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            _, col1 = st.columns([8, 0.8])
            with col1:
                if st.button("Next Step", type="primary", width="stretch"):
                    st.query_params["step"] = "cleaning"
                    st.rerun()