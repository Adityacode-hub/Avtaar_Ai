# import pandas as pd
# import numpy as np
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import joblib
# import matplotlib.pyplot as plt

# # 1. Load Dataset
# data_path = os.path.join('data', 'Cleaned_Auto_Insurance_Fraud_Claims.csv')
# df = pd.read_csv(data_path)

# print("✅ Dataset loaded. Shape:", df.shape)
# print("📌 Columns in the dataset:", df.columns.tolist())

# # 2. Basic Preprocessing
# df.dropna(inplace=True)
# print("🧹 Nulls removed. New shape:", df.shape)

# # 3. Identify fraud column automatically
# possible_fraud_columns = ['fraud_reported', 'fraud', 'isFraud', 'Fraud_found', 'Fraud_Ind']
# fraud_column = None

# for col in possible_fraud_columns:
#     if col in df.columns:
#         fraud_column = col
#         break

# if not fraud_column:
#     raise ValueError("❌ Could not find a column indicating fraud. Please check column names.")

# print(f"🔎 Fraud indicator column found: {fraud_column}")

# # 4. Encode categorical features
# le = LabelEncoder()
# for col in df.select_dtypes(include='object').columns:
#     if col != fraud_column:
#         df[col] = le.fit_transform(df[col])

# # 5. Define X and y
# X = df.drop(fraud_column, axis=1)

# # Convert target to numeric if needed
# if df[fraud_column].dtype == 'object':
#     y = df[fraud_column].map({'Y': 1, 'N': 0})
# else:
#     y = df[fraud_column]

# # 6. Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 7. Train Model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # 8. Evaluate
# y_pred = model.predict(X_test)
# print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
# print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# # 9. Feature Importances
# importances = model.feature_importances_
# features = X.columns

# print("\n🔍 Feature Importances:")
# for feature, importance in zip(features, importances):
#     print(f"{feature}: {importance:.4f}")

# # 10. Plot feature importances
# plt.figure(figsize=(10, 6))
# plt.title("Feature Importances")
# indices = np.argsort(importances)[::-1]
# plt.bar(range(X.shape[1]), importances[indices], align="center")
# plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
# plt.tight_layout()
# plt.show()

# # 11. Save Model
# os.makedirs('models', exist_ok=True)
# model_path = os.path.join('models', 'fraud_model.pkl')
# joblib.dump(model, model_path)
# print(f"\n💾 Model saved to {model_path}")


import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt

# ==== 1. Load Datasets ====
data_path = os.path.join('data', 'Cleaned_Auto_Insurance_Fraud_Claims.csv')
df = pd.read_csv(data_path)

# Optionally load and append new claims file (if it exists)
new_data_path = os.path.join('data', 'claims2.csv')
if os.path.exists(new_data_path):
    df_new = pd.read_csv(new_data_path)
    df = pd.concat([df, df_new], ignore_index=True)
    print(f"📥 claims2.csv found and appended. New dataset shape: {df.shape}")
else:
    print(f"📂 Loaded base dataset only. Shape: {df.shape}")

print("📌 Columns in the dataset:", df.columns.tolist())

# ==== 2. Basic Preprocessing ====
df.dropna(inplace=True)
print("🧹 Nulls removed. New shape:", df.shape)

# ==== 3. Detect Fraud Column ====
possible_fraud_columns = ['fraud_reported', 'fraud', 'isFraud', 'Fraud_found', 'Fraud_Ind']
fraud_column = next((col for col in possible_fraud_columns if col in df.columns), None)

if not fraud_column:
    raise ValueError("❌ No fraud indicator column found. Please check dataset.")

print(f"🔎 Using fraud indicator column: {fraud_column}")

# ==== 4. Encode Categorical Features ====
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    if col != fraud_column:
        df[col] = le.fit_transform(df[col])

# Save encoder
os.makedirs('models', exist_ok=True)
le_path = os.path.join('models', 'label_encoder.pkl')
joblib.dump(le, le_path)
print(f"💾 LabelEncoder saved to {le_path}")

# ==== 5. Define Features and Labels ====
X = df.drop(fraud_column, axis=1)

# Convert labels to 0/1 if necessary
if df[fraud_column].dtype == 'object':
    y = df[fraud_column].map({'Y': 1, 'N': 0})
else:
    y = df[fraud_column]

# ==== 6. Train/Test Split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 7. Train Model ====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==== 8. Evaluate Model ====
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))

# ==== 9. Feature Importances ====
importances = model.feature_importances_
features = X.columns

print("\n🔍 Feature Importances:")
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")

# ==== 10. Plot Importances ====
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
indices = np.argsort(importances)[::-1]
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# ==== 11. Save Model ====
model_path = os.path.join('models', 'fraud_model.pkl')
joblib.dump(model, model_path)
print(f"\n💾 Model saved to {model_path}")

# ==== 12. Predict on claims2.csv if needed ====
if os.path.exists(new_data_path):
    df_new_pred = pd.read_csv(new_data_path)
    df_new_pred.dropna(inplace=True)
    
    # Reload encoder
    le_loaded = joblib.load(le_path)
    for col in df_new_pred.select_dtypes(include='object').columns:
        df_new_pred[col] = le_loaded.fit_transform(df_new_pred[col])
    
    # Drop fraud column if it exists (for clean prediction)
    if fraud_column in df_new_pred.columns:
        df_new_pred.drop(columns=[fraud_column], inplace=True)

    model_loaded = joblib.load(model_path)
    preds = model_loaded.predict(df_new_pred)
    
    print("\n🧾 Predictions on claims2.csv:")
    print(preds)
