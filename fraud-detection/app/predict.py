# import pandas as pd
# import joblib

# # Load model and encoders
# def load_model(model_path='models/fraud_model.pkl', meta_path='models/model_metadata.pkl'):
#     model = joblib.load(model_path)
#     metadata = joblib.load(meta_path)
#     return model, metadata['encoders'], metadata['features']

# # Preprocess batch inputs using saved encoders
# def preprocess_batch(df, encoders, expected_features):
#     df = df.copy()
    
#     for col in df.columns:
#         if col in encoders:
#             try:
#                 df[col] = encoders[col].transform(df[col].astype(str))
#             except Exception as e:
#                 df[col] = df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)

#     # Ensure all expected features are present
#     missing_cols = set(expected_features) - set(df.columns)
#     for col in missing_cols:
#         df[col] = 0

#     return df[expected_features]
import pandas as pd
import joblib

def load_model(model_path='../models/fraud_model.pkl', meta_path='../models/model_metadata.pkl'):
    """Load the trained model and metadata from disk."""
    model = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return model, metadata['encoders'], metadata['features']

def preprocess_batch(df, encoders, expected_features):
    """Apply label encoding and align columns for prediction."""
    df = df.copy()

    for col in df.columns:
        if col in encoders:
            try:
                df[col] = encoders[col].transform(df[col].astype(str))
            except Exception:
                df[col] = df[col].map(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )

    # Fill missing columns with 0
    missing_cols = set(expected_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    return df[expected_features]

