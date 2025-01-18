import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Load datasets
data_path = "/home/nebius/data_unclean/NSCLC_cleaner_data.txt"
metadata_path = "/home/nebius/data_unclean/NSCLC_cleaner_metadata.txt"

# Step 1: Load Data
data = pd.read_csv(data_path, sep="\t")
metadata = pd.read_csv(metadata_path, sep="\t")

# Select Protein Data
sample_columns = data.columns[2:]  # Exclude 'Target' and 'Time point'
protein_data = data[sample_columns].T  # Transpose to get samples as rows
protein_data.columns = data["Target"]  # Set protein names as column names
protein_data.index.name = "Patient"

# Remove NaN values from protein data
protein_data = protein_data.dropna(axis=1)
print(f"Number of remaining proteins after removing NaNs: {protein_data.shape[1]}")

# Align Metadata with Protein Data
metadata["Patient"] = metadata["Patient"].str.strip()
protein_data.index = protein_data.index.str.strip()
metadata = metadata.set_index("Patient")
metadata = metadata.loc[protein_data.index]

# Extract features (X) and target variable (y)
X_original = protein_data.values
y_original = metadata["OS event (1= death)"].astype(int)

# Step 2: Feature Selection
k = 100  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=k)
X_selected = selector.fit_transform(X_original, y_original)
selected_features = protein_data.columns[selector.get_support()]

print(f"Selected top {k} features based on ANOVA F-statistics:")
print(selected_features)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_original, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Train and Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
}

results = []

# Random Classification Baseline
random_chance = max(np.mean(y_test), 1 - np.mean(y_test))
print(f"\nRandom Classification Chance: {random_chance:.4f}")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    # Store results in a structured format
    results.append({"Model": name, "Accuracy": acc, "ROC AUC": auc, "Random Chance": random_chance})
    print(f"{name} Results:")
    print(f"Accuracy: {acc}")
    print(f"ROC AUC: {auc}")
    print("Classification Report:")
    print(report)

# Save results to CSV
results_df = pd.DataFrame(results)
results_csv_path = "/home/nebius/data_unclean/model_comparison_results.csv"
results_df.to_csv(results_csv_path, index=False)

print(f"\nResults saved to {results_csv_path}")

