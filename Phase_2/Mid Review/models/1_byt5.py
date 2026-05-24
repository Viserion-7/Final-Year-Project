import pandas as pd

file_path = "sandhi_data.xlsx"

df = pd.read_excel(file_path)

print(df.head())
print(df.columns)
print("Total samples:", len(df))
