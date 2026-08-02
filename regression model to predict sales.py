import pandas as pd
import numpy as np

# 1. Load the cleaned dataset from Step 1
df = pd.read_csv('cleaned_sales_data.csv')
df.columns = df.columns.str.strip()

print("Columns available:", df.columns.tolist())

# 2. Choose features (X) and target (y)
feature_cols = ['Units Sold', 'Manufacturing Price', 'Sale Price', 'Discounts']
target_col = 'Sales'

X = df[feature_cols].values.astype(float)
y = df[target_col].values.astype(float)

# 3. Split into training and testing sets manually (80% train, 20% test)
np.random.seed(42)
indices = np.random.permutation(len(X))
split = int(len(X) * 0.8)
train_idx, test_idx = indices[:split], indices[split:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# 4. Add intercept column (bias term)
X_train_b = np.c_[np.ones(len(X_train)), X_train]
X_test_b = np.c_[np.ones(len(X_test)), X_test]

# 5. Train using Normal Equation: theta = (X^T X)^-1 X^T y
theta = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

# 6. Make predictions on test set
y_pred = X_test_b @ theta

# 7. Evaluate the model
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("\nModel Performance:")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)

# 8. Show feature coefficients
print("\nIntercept:", theta[0])
print("Feature coefficients:")
for feature, coef in zip(feature_cols, theta[1:]):
    print(feature, ":", round(coef, 2))

# 9. Predict sales for a sample input
sample = np.array([1, 1500, 50, 350, 1000])  # 1 for intercept
predicted_sales = sample @ theta
print("\nPredicted sales for sample input:", predicted_sales)