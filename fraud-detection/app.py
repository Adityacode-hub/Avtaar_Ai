# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# # Title
# st.title("🚨 Auto Insurance Fraud Detection")

# # Load the trained model
# model = joblib.load("fraud_model.pkl")

# # Load dataset just to get feature structure
# df = pd.read_csv("data/Cleaned_Auto_Insurance_Fraud_Claims.csv")

# # Get feature names
# target = 'Fraud_Ind'
# feature_columns = df.drop(columns=[target]).columns.tolist()

# # Initialize label encoders (same encoding logic used in training)
# encoders = {}
# for col in df.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col].astype(str))
#     encoders[col] = le

# # Streamlit form
# st.subheader("Enter Claim Details:")
# input_data = {}

# for col in feature_columns:
#     if df[col].dtype == 'int64' or df[col].dtype == 'float64':
#         input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))
#     else:
#         options = list(encoders[col].classes_)
#         input = st.selectbox(f"{col}", options)
#         input_data[col] = encoders[col].transform([input])[0]

# # Predict button
# if st.button("Predict Fraud"):
#     input_df = pd.DataFrame([input_data])
#     prediction = model.predict(input_df)[0]

#     if prediction == 1:
#         st.error("⚠️ This claim is predicted to be FRAUDULENT!")
#     else:
#         st.success("✅ This claim appears to be GENUINE.")


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

# === Title ===
st.title("🚨 Auto Insurance Fraud Detection")
st.markdown("Use manual form input OR upload a CSV to predict fraud.")

# === Load Model & Base Data ===
model = joblib.load("models/fraud_model.pkl")
df_base = pd.read_csv("data/Cleaned_Auto_Insurance_Fraud_Claims.csv")

target = 'Fraud_Ind'
feature_columns = df_base.drop(columns=[target]).columns.tolist()

# === Create LabelEncoders for consistency ===
encoders = {}
for col in df_base.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_base[col] = le.fit_transform(df_base[col].astype(str))
    encoders[col] = le

# === Sidebar: Choose Mode ===
mode = st.sidebar.radio("Choose Mode", ["📝 Manual Entry", "📁 Batch Prediction (CSV)"])

# =====================================================================================
# MODE 1: MANUAL ENTRY
# =====================================================================================
if mode == "📝 Manual Entry":
    st.subheader("Enter Claim Details:")
    input_data = {}

    for col in feature_columns:
        if df_base[col].dtype in ['int64', 'float64']:
            input_data[col] = st.number_input(f"{col}", value=float(df_base[col].mean()))
        else:
            options = list(encoders[col].classes_)
            selected = st.selectbox(f"{col}", options)
            input_data[col] = encoders[col].transform([selected])[0]

    if st.button("🔍 Predict Fraud (Manual Entry)"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️ This claim is predicted to be FRAUDULENT!")
        else:
            st.success("✅ This claim appears to be GENUINE.")

# =====================================================================================
# MODE 2: BATCH PREDICTION
# =====================================================================================
elif mode == "📁 Batch Prediction (CSV)":
    st.subheader("Upload CSV File for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV (e.g., claims2.csv)", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview", df_uploaded.head())

        # Preprocessing
        df_uploaded.dropna(inplace=True)

        # Drop fraud label column if present
        possible_fraud_cols = ['fraud_reported', 'fraud', 'isFraud', 'Fraud_found', 'Fraud_Ind']
        for col in possible_fraud_cols:
            if col in df_uploaded.columns:
                df_uploaded.drop(columns=[col], inplace=True)

        # Encode categorical features using the same encoders
        for col in df_uploaded.select_dtypes(include='object').columns:
            if col in encoders:
                try:
                    df_uploaded[col] = encoders[col].transform(df_uploaded[col].astype(str))
                except Exception as e:
                    st.warning(f"⚠️ Encoding issue with column '{col}': {e}")
            else:
                st.warning(f"⚠️ Unexpected categorical column '{col}' not seen during training. Skipped.")

        try:
            # Make predictions
            predictions = model.predict(df_uploaded)
            df_uploaded['fraud_prediction'] = predictions

            st.success("✅ Batch prediction completed!")
            st.write(df_uploaded.head())

            # Downloadable output
            csv_out = df_uploaded.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Predictions",
                data=csv_out,
                file_name="claims2_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# # === Title ===
# st.title("🚨 Auto Insurance Fraud Detection")
# st.markdown("Use manual input OR upload a CSV/Excel file to predict fraud.")

# # === Load model and training data ===
# model = joblib.load("models/fraud_model.pkl")
# df_base = pd.read_csv("data/Cleaned_Auto_Insurance_Fraud_Claims.csv")

# # === Define target and features ===
# target = 'Fraud_Ind'
# feature_columns = df_base.drop(columns=[target]).columns.tolist()

# # === Build encoders from training data ===
# encoders = {}
# for col in df_base.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     df_base[col] = le.fit_transform(df_base[col].astype(str))
#     encoders[col] = le

# # === Sidebar mode selection ===
# mode = st.sidebar.radio("Choose Mode", ["📝 Manual Entry", "📁 Batch Prediction"])

# # =============================================================================
# # 📝 MANUAL ENTRY MODE
# # =============================================================================
# if mode == "📝 Manual Entry":
#     st.subheader("Enter Claim Details Manually")

#     input_data = {}
#     for col in feature_columns:
#         if df_base[col].dtype in ['int64', 'float64']:
#             input_data[col] = st.number_input(f"{col}", value=float(df_base[col].mean()))
#         else:
#             options = list(encoders[col].classes_)
#             selected = st.selectbox(f"{col}", options)
#             input_data[col] = encoders[col].transform([selected])[0]

#     if st.button("🔍 Predict Fraud"):
#         input_df = pd.DataFrame([input_data])
#         prediction = model.predict(input_df)[0]

#         if prediction == 1:
#             st.error("⚠️ This claim is predicted to be FRAUDULENT!")
#         else:
#             st.success("✅ This claim appears to be GENUINE.")

# # =============================================================================
# # 📁 BATCH PREDICTION MODE
# # =============================================================================
# elif mode == "📁 Batch Prediction":
#     st.subheader("Upload CSV or Excel File for Batch Prediction")
#     uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

#     if uploaded_file is not None:
#         try:
#             # === Read uploaded file
#             if uploaded_file.name.endswith('.csv'):
#                 df_uploaded = pd.read_csv(uploaded_file)
#             elif uploaded_file.name.endswith('.xlsx'):
#                 df_uploaded = pd.read_excel(uploaded_file)
#             else:
#                 st.error("❌ Unsupported file format. Use CSV or Excel.")
#                 st.stop()

#             st.success(f"✅ File '{uploaded_file.name}' loaded successfully.")
#             st.write("📄 Preview of uploaded data:", df_uploaded.head())

#             # === Clean uploaded data
#             df_uploaded.dropna(inplace=True)

#             # Drop any fraud columns that may exist
#             fraud_cols = ['fraud_reported', 'fraud', 'isFraud', 'Fraud_found', 'Fraud_Ind']
#             df_uploaded.drop(columns=[col for col in fraud_cols if col in df_uploaded.columns], inplace=True)

#             # === Encode categorical columns
#             for col in df_uploaded.select_dtypes(include='object').columns:
#                 if col in encoders:
#                     df_uploaded[col] = encoders[col].transform(df_uploaded[col].astype(str))
#                 else:
#                     st.warning(f"⚠️ Unknown column '{col}' — skipped encoding.")

#             # === Predict and append results
#             predictions = model.predict(df_uploaded)
#             df_uploaded['fraud_prediction'] = predictions

#             st.success("✅ Predictions completed.")
#             st.write(df_uploaded.head())

#             # === Download result
#             csv = df_uploaded.to_csv(index=False).encode('utf-8')
#             st.download_button("⬇️ Download Results", data=csv, file_name="predictions.csv", mime="text/csv")

#         except Exception as e:
#             st.error(f"❌ Error: {e}")


# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder

# st.set_page_config(page_title="Auto Insurance Fraud Detection", layout="wide")

# st.title("🚨 Auto Insurance Fraud Detection")

# # Load model and reference data
# model = joblib.load("fraud_model.pkl")
# df_ref = pd.read_csv("data/Cleaned_Auto_Insurance_Fraud_Claims.csv")

# # Define features and target
# target = "Fraud_Ind"
# features = df_ref.drop(columns=[target]).columns.tolist()

# # Fit encoders from reference data
# encoders = {}
# for col in df_ref.select_dtypes(include='object').columns:
#     le = LabelEncoder()
#     df_ref[col] = le.fit_transform(df_ref[col].astype(str))
#     encoders[col] = le

# # --- SINGLE PREDICTION SECTION ---
# st.header("🔍 Single Claim Prediction")

# input_data = {}
# with st.form("single_form"):
#     for col in features:
#         if col in encoders:
#             options = list(encoders[col].classes_)
#             selected = st.selectbox(f"{col}", options)
#             input_data[col] = encoders[col].transform([selected])[0]
#         elif "date" in col.lower():
#             date_val = st.date_input(f"{col}")
#             input_data[col] = pd.to_datetime(date_val).timestamp()
#         else:
#             input_data[col] = st.number_input(f"{col}", value=float(df_ref[col].mean()))

#     submitted = st.form_submit_button("Predict Fraud")

# if submitted:
#     input_df = pd.DataFrame([input_data])
#     pred = model.predict(input_df)[0]
#     if pred == 1:
#         st.error("⚠️ This claim is predicted to be FRAUDULENT!")
#     else:
#         st.success("✅ This claim appears to be GENUINE.")

# # --- BATCH PREDICTION SECTION ---
# st.header("📤 Batch Upload for Multiple Predictions")

# uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# if uploaded_file:
#     try:
#         if uploaded_file.name.endswith("csv"):
#             df_uploaded = pd.read_csv(uploaded_file)
#         else:
#             df_uploaded = pd.read_excel(uploaded_file)

#         df_result = df_uploaded.copy()

#         # Align with training features
#         aligned_df = pd.DataFrame(columns=features)

#         for col in features:
#             if col not in df_uploaded.columns:
#                 st.warning(f"⚠️ Missing column '{col}' — filled with default")
#                 aligned_df[col] = df_ref[col].mean()
#             else:
#                 aligned_df[col] = df_uploaded[col]

#         # Preprocessing (encoding and date conversion)
#         for col in aligned_df.columns:
#             if col in encoders:
#                 try:
#                     aligned_df[col] = encoders[col].transform(aligned_df[col].astype(str))
#                 except ValueError as e:
#                     st.error(f"❌ Unseen label in column '{col}': {e}")
#                     st.stop()
#             elif "date" in col.lower():
#                 try:
#                     aligned_df[col] = pd.to_datetime(aligned_df[col], errors="coerce").astype(int) / 10**9
#                 except Exception as e:
#                     st.warning(f"⚠️ Failed to convert date in '{col}': {e}")
#                     aligned_df[col] = 0  # default fallback

#         # Predict
#         predictions = model.predict(aligned_df)
#         df_result["Fraud_Prediction"] = predictions

#         st.subheader("🔎 Prediction Results")
#         st.write(df_result.head())

#         # Download button
#         csv = df_result.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇️ Download Results", csv, "fraud_predictions.csv", "text/csv")

#     except Exception as e:
#         st.error(f"❌ Error: {e}")
