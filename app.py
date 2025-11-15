import streamlit as st
import pandas as pd
from pipeline import run_pipeline

st.set_page_config(page_title="Marketing Pipeline", layout="wide")

st.title("📊 Marketing Campaign Feature Engineering")

st.write("Upload a CSV file to run the full cleaning + feature engineering pipeline.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("📥 Uploaded Data")
    st.write(df.head())

    # Run pipeline
    with st.spinner("Running pipeline..."):
        result = run_pipeline(df)

    st.success("Pipeline completed!")

    st.subheader("📤 Processed Output (first 5 rows)")
    st.write(result)

    # Download processed CSV
    st.download_button(
        label="⬇️ Download Processed CSV",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="processed_output.csv",
        mime="text/csv",
    )
