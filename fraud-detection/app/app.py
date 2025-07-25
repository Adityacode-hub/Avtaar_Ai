import streamlit as st
import pandas as pd
from predict import load_model, preprocess_batch

# Load model and encoders
model, encoders, feature_columns = load_model()

st.set_page_config(page_title="Insurance Fraud Detection", layout="wide")
st.title("🕵️ Insurance Fraud Detection App")

st.markdown("Upload a CSV or Excel file to make batch predictions.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload File", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            batch_df = pd.read_csv(uploaded_file)
        else:
            batch_df = pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(batch_df)

        # Preprocess
        preprocessed = preprocess_batch(batch_df, encoders, feature_columns)
        predictions = model.predict(preprocessed)

        # Append predictions to original DataFrame
        batch_df['Prediction'] = predictions
        batch_df['Prediction_Label'] = batch_df['Prediction'].map({1: 'Fraud', 0: 'Genuine'})

        st.success("✅ Batch prediction completed!")
        st.dataframe(batch_df)

        # Download button
        csv_download = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv_download, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
