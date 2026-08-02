import pandas as pd

# 1. Load the dataset
df = pd.read_csv('data (1).csv')

# 2. Explore the data
print("First 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Handle missing values
# Fill missing numeric values with the column's average
df['Duration'] = df['Duration'].fillna(df['Duration'].mean())
df['Pulse'] = df['Pulse'].fillna(df['Pulse'].mean())
df['Maxpulse'] = df['Maxpulse'].fillna(df['Maxpulse'].mean())
df['Calories'] = df['Calories'].fillna(df['Calories'].mean())

# Fix the Date column (this dataset often has one broken date value)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# 4. Filter rows — remove rows with unrealistic/duplicate data
df = df.drop_duplicates()
df = df[df['Duration'] > 0]  # remove invalid duration entries

# 5. Create a new column — Calories burned per minute
df['Calories_per_min'] = df['Calories'] / df['Duration']

# 6. Final check
print("\nCleaned dataset shape:", df.shape)
print(df.head())

# 7. Save the cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
print("\nSaved as cleaned_data.csv")