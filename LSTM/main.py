import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import MinMaxScaler
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the Multi-Step Stacked LSTM Model
class StackedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=2, dropout=0.2):
        super(StackedLSTM, self).__init__()
        # Stacked LSTM with multiple layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Output at the last time step
        out = self.fc(out)
        return out


# Parameters
input_size = 25
hidden_size = 50
output_size = 3
num_layers = 2  # Stacked LSTM with 2 layers
num_epochs = 20
batch_size = 64
learning_rate = 0.001
seq_length = 60

# Load and preprocess the CSV data
file_path = 'data/merged_output_pcib.csv'
data = pd.read_csv(file_path)

# Select relevant columns for input features and target
features = ['close', 'open', 'high', 'low', 'Volume', 'Plot1', 'Plot2', 'Plot3', 'RSI', 'Histogram', 'MACD', 'Signal',
            'Volume MA', 'post_volume', 'max_content_length', 'min_content_length', 'average_post_length',
            'num_bullish', 'num_neutral', 'num_bearish', 'unique_posters', 'max_likes', 'min_likes', 'average_likes',
            'like_rate']
target = 'close'

# Handle missing values
data.fillna(0, inplace=True)

# Separate feature and target data
feature_data = data[features]
target_data = data[[target]]

# Scale features and target separately
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Fit scaler on features and transform
feature_data = feature_scaler.fit_transform(feature_data)
# Fit scaler on target and transform
target_data = target_scaler.fit_transform(target_data)

# Prepare sequences and targets for multi-step forecasting
x_data = []
y_data = []

for i in range(len(feature_data) - seq_length - 10):
    x_data.append(feature_data[i:i + seq_length])
    y_data.append([
        target_data[i + seq_length][0],
        target_data[i + seq_length + 4][0],
        target_data[i + seq_length + 9][0]
    ])

# Convert lists to PyTorch tensors
x_data = torch.tensor(x_data, dtype=torch.float32)
y_data = torch.tensor(y_data, dtype=torch.float32)

# Move data to the device (GPU if available)
x_data = x_data.to(device)
y_data = y_data.to(device)

# Create TensorDataset and split into train, validation, and test sets
dataset = TensorDataset(x_data, y_data)
num_samples = len(x_data)
train_size = int(0.65 * num_samples)
val_size = int(0.20 * num_samples)
test_size = num_samples - train_size - val_size

train_dataset = Subset(dataset, range(0, train_size))
val_dataset = Subset(dataset, range(train_size, train_size + val_size))
test_dataset = Subset(dataset, range(train_size + val_size, num_samples))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model, define the loss function and optimizer
model = StackedLSTM(input_size, hidden_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/stacked_lstm_experiment')

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val, y_val = x_val.to(device), y_val.to(device)
            val_outputs = model(x_val)
            val_loss += criterion(val_outputs, y_val).item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    # Log training and validation loss to TensorBoard
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

print("Training Complete")

# Close TensorBoard writer
writer.close()

# Example prediction with inverse transform to original scale
model.eval()
sample_input = x_data[0].unsqueeze(0)
predicted_scaled = model(sample_input)
predicted_prices = target_scaler.inverse_transform(predicted_scaled.cpu().detach().numpy())
print("Predicted closing prices (1-step, 5-step, 10-step):", predicted_prices)

# Testing with inverse transform to original scale
test_loss = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        x_test, y_test = x_test.to(device), y_test.to(device)
        test_outputs = model(x_test)
        test_loss += criterion(test_outputs, y_test).item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")

model_path = 'stacked_lstm_model.pth'
torch.save(model.state_dict(), model_path)
print("Model saved to", model_path)
