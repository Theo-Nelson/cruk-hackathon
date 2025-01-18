import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data_path = "/home/nebius/new_project_/data/input.txt"  # Adjust path as needed
df = pd.read_csv(data_path, sep="\t")  # Adjust separator if needed

# Define features and target
target_col = "resp"  # Change to the correct column name for IC50 values
X = df.drop(columns=[target_col, "drug", "cell"])  # Drop irrelevant columns
y = df[target_col]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to test
models = {
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "ElasticNet": ElasticNet(),
    "RandomForest": RandomForestRegressor(),
    "SupportVectorRegressor": SVR(),
    "XGBoost": XGBRegressor(tree_method="hist", n_jobs=-1),
}

# Store results
results = []

# Test each model
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": name, "MSE": mse, "R2": r2})
    
    # Plot scatterplot of predicted vs actual values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.title(f"{name}: Predicted vs Actual IC50")
    plt.xlabel("Actual IC50")
    plt.ylabel("Predicted IC50")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name}_scatterplot.png")
    plt.show()

# Output results
results_df = pd.DataFrame(results)
print(results_df)
