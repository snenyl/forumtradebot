import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from backtesting import Backtest, Strategy

# Load and preprocess the data
data = pd.read_csv('data/merged_output_pcib.csv', parse_dates=True, index_col='time')
data.index = pd.to_datetime(data.index, unit='s')
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data.dropna(inplace=True)

# Select features for LSTM input and scale them
features = ['Close', 'Open', 'High', 'Low', 'Volume', 'Plot1', 'Plot2', 'Plot3', 'RSI', 'Histogram', 'MACD', 'Signal',
            'Volume MA', 'post_volume', 'max_content_length', 'min_content_length', 'average_post_length',
            'num_bullish', 'num_neutral', 'num_bearish', 'unique_posters', 'max_likes', 'min_likes',
            'average_likes', 'like_rate']
feature_data = data[features]
feature_scaler = MinMaxScaler()
feature_data_scaled = feature_scaler.fit_transform(feature_data)

# Load the trained LSTM model
input_size = len(features)
hidden_size = 50
output_size = 3  # Predicting 1, 5, and 10 steps ahead
num_layers = 1
seq_length = 120

# Define the LSTM model architecture
class MultiStepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=3, num_layers=1):
        super(MultiStepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Predict 3 future closing prices

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Output at the last time step
        out = self.fc(out)
        return out

# Instantiate and load model
model = MultiStepLSTM(input_size, hidden_size, output_size, num_layers)
model.load_state_dict(torch.load('lstm_model.pth', map_location=torch.device('cpu')))  # Load model safely
model.eval()

# Generate LSTM predictions for backtesting
target_scaler = MinMaxScaler()
target_scaler.fit(data[['Close']])
predictions_1, predictions_5, predictions_10 = [], [], []

for i in range(seq_length, len(feature_data_scaled) - 10):
    sequence = torch.tensor(feature_data_scaled[i - seq_length:i], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(sequence).squeeze().numpy()

    # Extract each prediction (1-step, 5-step, and 10-step) and scale back to original
    pred_1 = target_scaler.inverse_transform([[pred[0]]])[0][0]
    pred_5 = target_scaler.inverse_transform([[pred[1]]])[0][0]
    pred_10 = target_scaler.inverse_transform([[pred[2]]])[0][0]

    predictions_1.append(pred_1)
    predictions_5.append(pred_5)
    predictions_10.append(pred_10)

# Align predictions with the data for backtesting
data = data.iloc[seq_length + 10:]
data['Predicted_Close_1'] = predictions_1
data['Predicted_Close_5'] = predictions_5
data['Predicted_Close_10'] = predictions_10

# Define the strategy based on direct predictions
class LstmPredictionStrategy(Strategy):
    short_window = 20  # Short SMA period
    long_window = 50   # Long SMA period
    equity_percentage = 0.05  # Use 5% of account equity for each trade

    def init(self):
        # Indicators for each prediction horizon
        self.predicted_close_5 = self.I(lambda: self.data.Predicted_Close_5, name='Predicted_Close_5')

        # Define short and long SMAs
        self.sma_short = self.I(pd.Series(self.data.Close).rolling(self.short_window).mean, name='SMA_Short')
        self.sma_long = self.I(pd.Series(self.data.Close).rolling(self.long_window).mean, name='SMA_Long')

    def next(self):
        # Use the 5-step prediction to determine trend direction
        pred_close_5 = self.predicted_close_5[-1]
        current_close = self.data.Close[-1]

        # Calculate position size as a percentage of account equity
        position_value = self.equity * self.equity_percentage
        size = position_value / current_close  # Size in units of the asset

        # Ensure minimum size of 1 for rounding errors
        size = max(int(size), 1)

        # SMA values for current time step
        sma_short = self.sma_short[-1]
        sma_long = self.sma_long[-1]

        # Buy if prediction is higher than current close and SMA short crosses above SMA long
        if pred_close_5 > current_close and sma_short > sma_long:
            if not self.position.is_long:
                self.buy(size=size)

        # Sell if prediction is lower than current close and SMA short crosses below SMA long
        elif pred_close_5 < current_close and sma_short < sma_long:
            if not self.position.is_short:
                self.sell(size=size)

# Run the backtest
bt = Backtest(data, LstmPredictionStrategy, cash=10000, commission=0.002)
stats = bt.run()
print(stats)
bt.plot()
