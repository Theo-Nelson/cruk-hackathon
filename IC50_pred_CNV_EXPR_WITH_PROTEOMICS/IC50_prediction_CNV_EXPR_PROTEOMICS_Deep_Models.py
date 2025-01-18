import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define datasets and file paths
file_paths = {
    "input.txt": "input.txt",  # Update with the actual path
    "merged_input_with_proteomics.txt": "merged_input_with_proteomics.txt",
}

# Define a complex architecture
class ComplexModel(nn.Module):
    def __init__(self, input_dim):
        super(ComplexModel, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.rnn = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.unsqueeze(1)  # Add sequence dimension for LSTM
        _, (hn, _) = self.rnn(x)
        hn = hn[-2:].transpose(0, 1).contiguous().view(hn.shape[1], -1)  # Concatenate bidirectional outputs
        return self.fc(hn)

# Training and evaluation function
def train_and_evaluate(dataset_name, data_path):
    print(f"Training and evaluating on {dataset_name}...")

    # Load data
    df = pd.read_csv(data_path, sep="\t")
    target_col = "resp"  # Target column

    X = df.drop(columns=[target_col, "drug", "cell"], errors="ignore")
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    model = ComplexModel(input_dim=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 100
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))

        # Evaluate on test data
        model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.cpu().numpy())

        test_losses.append(test_loss / len(test_loader))

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_losses[-1]:.4f} | Test Loss: {test_losses[-1]:.4f}")

    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    print(f"R^2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    # Plot predicted vs actual
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'k--', lw=2)
    plt.title(f"{dataset_name}: Predicted vs Actual IC50")
    plt.xlabel("Actual IC50")
    plt.ylabel("Predicted IC50")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_scatterplot.png")
    plt.show()

    return r2, mse, mae

# Compare datasets
results = {}
for dataset_name, data_path in file_paths.items():
    r2, mse, mae = train_and_evaluate(dataset_name, data_path)
    results[dataset_name] = {
        "R^2": r2,
        "MSE": mse,
        "MAE": mae
    }

# Print results
for dataset, metrics in results.items():
    print(f"Results for {dataset}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


