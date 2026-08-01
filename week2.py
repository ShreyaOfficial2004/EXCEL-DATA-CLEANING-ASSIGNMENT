import pandas as pd

df = pd.read_csv("SQL_Sales_Dataset_200_Rows.xlsx - Sheet1.csv")

print(df.head())
print(df.info())
print(df.describe())