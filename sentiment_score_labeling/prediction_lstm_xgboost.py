import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from clearml import Task
import xgboost as xgb
import plotly.graph_objects as go
from itertools import product

# Initialize ClearML task
task = Task.init(project_name="Stock Prediction", task_name="LSTM with ClearML grid search")
task.connect({"disable_gpu_monitoring": True})  # Disable GPU monitoring explicitly

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
# target = "close_pct_change"
target = "close_pct_change_5d"

# Shift the target to predict the next day's change
df["next_close_pct_change"] = df[target].shift(-1)
df = df.dropna()  # Drop rows with NaN values

# Normalize features
df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())

# Split data into training and testing
X = df[features]
y = df["next_close_pct_change"]

# Split data into training, validation, and testing sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2 of the total

# Debugging split sizes
print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Prepare data for LSTM (3D input: samples, timesteps, features)
def create_sequences(data, target, sequence_length=10):
    X_seq, y_seq = [], []
    for i in range(len(data) - sequence_length):
        X_seq.append(data[i:i+sequence_length])
        y_seq.append(target[i+sequence_length])
    return np.array(X_seq), np.array(y_seq)

sequence_length = 10
X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val.values, y_val.values, sequence_length)
X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, sequence_length)

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
val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])  # Use the last hidden state
        return out

# Grid Search Hyperparameters
lstm_hidden_dims = [32, 64]
learning_rates = [0.001, 0.01]
xgboost_params_grid = {
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
    "n_estimators": [50, 100]
}

best_mse = float("inf")
best_config = None

# Initialize ClearML logger
logger = task.get_logger()

# Perform Grid Search
for hidden_dim, lr in product(lstm_hidden_dims, learning_rates):
    # Initialize LSTM Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train LSTM
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)

            # Ensure outputs and targets have the same shape
            outputs = outputs.squeeze(-1)  # Squeeze only the last dimension
            if outputs.dim() == 0:  # Handle single-element batch
                outputs = outputs.unsqueeze(0)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate LSTM
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs.squeeze(), targets).item()

        val_loss /= len(val_loader)

        # Log to ClearML
        logger.report_scalar("LSTM Loss", f"Train (hidden_dim={hidden_dim}, lr={lr})", iteration=epoch, value=train_loss)
        logger.report_scalar("LSTM Loss", f"Validation (hidden_dim={hidden_dim}, lr={lr})", iteration=epoch, value=val_loss)

    # Generate LSTM Residuals
    model.eval()
    lstm_predictions = []
    with torch.no_grad():
        for sequences, _ in test_loader:
            sequences = sequences.to(device)
            preds = model(sequences)

            # Ensure preds maintains the correct shape
            preds = preds.squeeze(-1) if preds.dim() > 1 else preds.unsqueeze(0)  # Handle single-element batch
            lstm_predictions.extend(preds.cpu().numpy())



    residuals = y_val_seq[:len(lstm_predictions)] - np.array(lstm_predictions)

    # Combine Residuals with Features for XGBoost
    X_train_xgb = np.hstack([X_train_seq.reshape(len(X_train_seq), -1), y_train_seq.reshape(-1, 1)])
    X_val_xgb = np.hstack([X_val_seq[:len(residuals)].reshape(len(residuals), -1), residuals.reshape(-1, 1)])
    dtrain = xgb.DMatrix(X_train_xgb, label=y_train_seq)
    dval = xgb.DMatrix(X_val_xgb, label=y_val_seq[:len(residuals)])

    # Perform Grid Search for XGBoost
    for xgb_params in product(*xgboost_params_grid.values()):
        params = dict(zip(xgboost_params_grid.keys(), xgb_params))
        params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

        # Train XGBoost
        xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=50,
            evals=[(dval, "validation")],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # Evaluate XGBoost
        xgb_predictions = xgb_model.predict(dval)
        mse = mean_squared_error(y_val_seq[:len(xgb_predictions)], xgb_predictions)

        # Log Results
        logger.report_scalar("XGBoost MSE", f"LSTM={hidden_dim}, lr={lr}, XGBoost={params}", iteration=0, value=mse)

        # Track Best Model
        if mse < best_mse:
            best_mse = mse
            best_config = {"hidden_dim": hidden_dim, "lr": lr, "xgb_params": params}

print(f"Best Configuration: {best_config}, MSE: {best_mse}")

# Model, Loss, Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_dim=X_train.shape[1], hidden_dim=64, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop with Validation and ClearML Logging
num_epochs = 60
for epoch in range(num_epochs):
    # Training phase
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

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for sequences, targets in val_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            val_loss += criterion(outputs.squeeze(), targets).item()

    val_loss /= len(val_loader)

    # Log training and validation loss to ClearML
    logger.report_scalar("Loss", "train", iteration=epoch, value=train_loss)
    logger.report_scalar("Loss", "validation", iteration=epoch, value=val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


# Evaluate the LSTM Model
model.eval()
lstm_predictions = []
with torch.no_grad():
    for sequences, _ in test_loader:
        sequences = sequences.to(device)
        preds = model(sequences)
        # print(f"Preds shape before squeeze: {preds.shape}")

        # Avoid collapsing all dimensions for single-element batches
        if preds.dim() == 2 and preds.shape[1] == 1:
            preds = preds.squeeze(1)  # Squeeze only the last dimension (keep batch dimension)
        elif preds.dim() == 1:
            preds = preds  # Already squeezed properly
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        # print(f"Preds shape after squeeze: {preds.shape}")
        lstm_predictions.extend(preds.cpu().numpy())

residuals = y_test_seq - np.array(lstm_predictions[:len(y_test_seq)])

# Combine residuals with original features for XGBoost
X_train_xgb = np.hstack([X_train_seq.reshape(len(X_train_seq), -1), y_train_seq.reshape(-1, 1)])
X_val_xgb = np.hstack([X_val_seq.reshape(len(X_val_seq), -1), y_val_seq.reshape(-1, 1)])
X_test_xgb = np.hstack([X_test_seq.reshape(len(X_test_seq), -1), residuals.reshape(-1, 1)])

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train_xgb, label=y_train_seq)
dval = xgb.DMatrix(X_val_xgb, label=y_val_seq)
dtest = xgb.DMatrix(X_test_xgb, label=y_test_seq)

# Define XGBoost parameters
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

# Train the XGBoost model with validation monitoring
xgb_model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dval, "validation"), (dtest, "test")],
    early_stopping_rounds=10,
)

# Make predictions with XGBoost
xgb_predictions = xgb_model.predict(dtest)

# Combine LSTM and XGBoost predictions
final_predictions = np.array(lstm_predictions[:len(xgb_predictions)]) + xgb_predictions

# Evaluate the final model
final_mse = mean_squared_error(y_test_seq[:len(final_predictions)], final_predictions)
print(f"Mean Squared Error (Combined LSTM + XGBoost): {final_mse:.4f}")

# Plot results with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test_seq[:len(final_predictions)], mode='lines', name='True Values'))
fig.add_trace(go.Scatter(y=final_predictions, mode='lines', name='Combined Predictions'))
fig.update_layout(
    title="True vs Combined Predictions (LSTM + XGBoost)",
    xaxis_title="Index",
    yaxis_title="Percentage Change",
    legend_title="Legend",
    # template="plotly_dark"
)
fig.show()


# Plot features used

# Get feature importance from XGBoost
feature_importance = xgb_model.get_score(importance_type="weight")

# Feature names for XGBoost
feature_names_xgb = [f"seq_{i}" for i in range(X_train_seq.shape[1] * X_train_seq.shape[2])] + ["residual"]

# Map XGBoost feature names (e.g., f0, f1) to actual feature names
mapped_features = {f"f{i}": feature_names_xgb[i] for i in range(len(feature_names_xgb))}

# Convert feature importance keys to actual feature names
sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
feature_names, importance_values = zip(*[(mapped_features[k], v) for k, v in sorted_features])

# Plot feature importance
fig = go.Figure()
fig.add_trace(go.Bar(
    x=importance_values,
    y=feature_names,
    orientation='h',
    name='Feature Importance'
))
fig.update_layout(
    title="Feature Importance in XGBoost Model",
    xaxis_title="Importance Score",
    yaxis_title="Features",
    # template="plotly_dark",
    height=600
)
fig.show()
