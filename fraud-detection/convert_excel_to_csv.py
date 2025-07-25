import pandas as pd
import os

# Set path to Excel file
excel_path = os.path.join('data', 'Cleaned_Auto_Insurance_Fraud_Claims.xlsx')
csv_path = os.path.join('data', 'insurance_claims.csv')

# Read and convert
df = pd.read_excel(excel_path)
df.to_csv(csv_path, index=False)

print("✅ Excel converted to CSV successfully!")
