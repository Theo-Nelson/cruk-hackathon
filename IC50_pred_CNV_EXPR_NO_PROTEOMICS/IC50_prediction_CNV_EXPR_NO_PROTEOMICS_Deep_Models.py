import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
data_path = "/home/nebius/new_project_/data/input.txt"  # Adjust path as needed
df = pd.read_csv(data_path, sep="\t")  # Adjust delimiter if needed

# Define features and target
target_col = "resp"  # Adjust if the column name for IC50 is different
X = df.drop(columns=[target_col, "drug", "cell"])  # Drop irrelevant columns
y = df[target_col]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors and move to the appropriate device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

# Create PyTorch datasets and loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Models
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(input_dim // 4 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)


class RNN(nn.Module):
    def __init__(self, input_dim):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        _, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])


# Training function
def train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_losses = []
    test_losses = []

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
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")

    return train_losses, test_losses


# Evaluate model and plot results
def evaluate_and_plot(model, test_loader, y_test, model_name):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())  # Move to CPU for plotting
    predictions = np.vstack(predictions)

    # Calculate R^2 score
    r2 = r2_score(y_test.cpu().numpy(), predictions)
    print(f"{model_name} R^2 Score: {r2:.4f}")

    # Plot predicted vs actual values
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test.cpu().numpy(), predictions, alpha=0.6)
    plt.title(f"{model_name}: Predicted vs Actual IC50 (R^2: {r2:.4f})")
    plt.xlabel("Actual IC50")
    plt.ylabel("Predicted IC50")
    plt.plot([y_test.min().cpu(), y_test.max().cpu()], [y_test.min().cpu(), y_test.max().cpu()], 'k--', lw=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name}_scatterplot.png")
    plt.show()


# Run experiments
input_dim = X_train.shape[1]
models = {
    "SimpleNN": SimpleNN(input_dim).to(device),
    "CNN": CNN(input_dim).to(device),
    "RNN": RNN(input_dim).to(device),
}

for model_name, model in models.items():
    print(f"Training {model_name}...")
    train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=50)
    evaluate_and_plot(model, test_loader, y_test_tensor, model_name)

