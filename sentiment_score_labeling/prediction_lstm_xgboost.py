import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from clearml import Task
import plotly.express as px

# Initialize ClearML task
task = Task.init(project_name="Stock Prediction", task_name="LSTM with ClearML")

# Load the data
file_path = "combined_metrics_and_stock_data.parquet"
df = pd.read_parquet(file_path)

print(df.columns)

# Features and target
features = [
    "Volume", "Histogram", "Signal", "MACD", "credibility_value",
    "outlook_value", "referential_depth_value", "sentiment_value",
    "z_score_scaled", "max_words_normalized"
]
target = "close_pct_change_5d"

# Shift the target to predict the next day's change
df["next_close_pct_change"] = df[target].shift(-1)
df = df.dropna()  # Drop rows with NaN values

# Normalize features
df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

# Split data into training and testing
X = df[features]
y = df["next_close_pct_change"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare data for LSTM (3D input: samples, timesteps, features)
def create_sequences(data, target, sequence_length=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        X_seq.append(data[i:i+sequence_length])
        y_seq.append(target[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, sequence_length)

# Debugging shapes
print(f"X_train_seq shape: {X_train_seq.shape}")
print(f"y_train_seq shape: {y_train_seq.shape}")
print(f"X_test_seq shape: {X_test_seq.shape}")
print(f"y_test_seq shape: {y_test_seq.shape}")

# Create Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
test_dataset = TimeSeriesDataset(X_test_seq, y_test_seq)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with ClearML Logging
num_epochs = 60
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    # Log to ClearML
    task.get_logger().report_scalar("Loss", "train", iteration=epoch, value=train_loss)
    task.get_logger().report_scalar("Loss", "val", iteration=epoch, value=val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")



# Evaluate the Model
model.eval()
lstm_predictions = []

with torch.no_grad():
    for sequences, _ in test_loader:
        sequences = sequences.to(device)
        preds = model(sequences)  # Output: torch.Size([8, 1])

        # Ensure correct shape for iteration
        if preds.dim() > 1 and preds.shape[-1] == 1:  # Handle [batch_size, 1]
            lstm_predictions.extend(preds.squeeze(-1).cpu().numpy())  # Squeeze only the last dimension
        elif preds.dim() == 1:  # Already in [batch_size] shape
            lstm_predictions.extend(preds.cpu().numpy())
        else:  # Handle unexpected shape
            raise ValueError(f"Unexpected shape of preds: {preds.shape}")

print(f"Collected LSTM predictions: {len(lstm_predictions)}")
# Calculate residuals
residuals = y_test_seq - np.array(lstm_predictions[:len(y_test_seq)])

# Print Evaluation Metric
mse = mean_squared_error(y_test_seq[:len(lstm_predictions)], lstm_predictions)
print(f"Mean Squared Error (LSTM): {mse:.4f}")
