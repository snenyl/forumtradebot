from pymongo import MongoClient
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# MongoDB connection parameters
mongo_uri = "mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT"
database_name = "osebx"
collection_name = "ULTI_1H"

# Connect to MongoDB and retrieve data
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

# Define your query, if any (e.g., fetching the latest data)
query = {}  # Adjust your query if needed
cursor = collection.find(query)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(list(cursor))

df['time'] = pd.to_datetime(df['time'])  # Ensure Date is in datetime format
df.set_index('time', inplace=True)  # Set Date as the index

df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})

# df = df[df.index >= '2023-01-01'] # Need to scale down at higher resolutions.

# Close the MongoDB connection
client.close()

print(df.columns)
print(df.head)

# Function to calculate RSI
def calculate_rsi(prices, period=7):
    # Convert to pandas Series if needed
    prices = pd.Series(prices)
    delta = prices.diff()  # Calculate the daily change in prices
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Avg gain over period
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Avg loss over period
    rs = gain / loss  # Relative Strength
    rsi = 100 - (100 / (1 + rs))  # RSI calculation
    return rsi


# Define an RSI-based trading strategy
class RSIStrategy(Strategy):
    rsi_period = 14  # Default RSI period (can be adjusted)
    oversold_level = 45  # RSI level considered oversold (buy)
    overbought_level = 55  # RSI level considered overbought (sell)
    custom_size=100

    def init(self):
        # Compute RSI indicator
        self.rsi = self.I(calculate_rsi, self.data.Close, self.rsi_period)

    def next(self):
        if self.rsi[-1] < self.oversold_level:  # RSI crosses below 30
            if not self.position:  # Open a position if not already in one
                self.buy(size=self.custom_size)
        elif self.rsi[-1] > self.overbought_level:  # RSI crosses above 70
            if self.position:  # Close position if RSI indicates overbought
                self.sell(size=self.custom_size)

# Run backtest
bt = Backtest(df, RSIStrategy, cash=10000, commission=.002)
stats = bt.run()
bt.plot()