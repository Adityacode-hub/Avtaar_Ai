import pandas as pd

df = pd.read_csv("data/Cleaned_Auto_Insurance_Fraud_Claims.csv")
print("📌 Columns in your dataset:")
print(df.columns.tolist())
