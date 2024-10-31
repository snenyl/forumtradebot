import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# Load main data and post volume data
data = pd.read_csv('data/OSL_DLY ULTI, 1H.csv', parse_dates=True, index_col='time')
data.index = pd.to_datetime(data.index, unit='s')
data = data[data.index >= '2019-02-25']

# Rename columns for backtesting
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(axis=1, how='all')

# Load post volume data and merge
post_volume = pd.read_csv('data/post_volume_by_hour_ulti.csv', parse_dates=['post_published_date'])
post_volume.set_index('post_published_date', inplace=True)
data = data.merge(post_volume, left_index=True, right_index=True, how='left')
data['post_volume'].fillna(0, inplace=True)


# Define strategy with post_volume as a custom indicator
class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

        # Add post_volume as an indicator for plotting
        self.post_volume_indicator = self.I(lambda: self.data.post_volume, name='Post Volume')

    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()
        elif crossover(self.sma2, self.sma1):
            self.sell()


class MonthlyBuyHold(Strategy):
    def init(self):
        self.last_trade_month = None  # To track the last month when a trade was made

    def next(self):
        # Get the current date of the data
        current_date = self.data.index[-1]
        current_month = current_date.month
        current_day = current_date.day

        # Check if it's the 20th day of a new month to execute a buy
        if self.last_trade_month != current_month and current_day == 20:
            # Buy on the 20th day of each new month
            self.buy()
            # Update last trade month to the current month
            self.last_trade_month = current_month


class MonthlyBuyHoldWithStoploss(Strategy):
    def init(self):
        self.last_trade_month = None  # To track the last month when a trade was made
        self.stop_price = None  # Track the stop-loss price

    def next(self):
        # Get the current date of the data
        current_date = self.data.index[-1]
        current_month = current_date.month
        current_day = current_date.day

        # Check if it's the 20th day of a new month to execute a buy
        if self.last_trade_month != current_month and current_day == 20:
            # Buy on the 20th day of each new month
            self.buy()
            # Update last trade month to the current month
            self.last_trade_month = current_month
            # Set stop-loss price to 5% below the current price
            self.stop_price = self.data.Close[-1] * 0.90  # 5% below purchase price

        # Check if current price has hit the stop-loss
        elif self.stop_price and self.data.Close[-1] <= self.stop_price:
            # Sell 50% of current holdings if stop-loss is hit
            units_to_sell = int(0.5 * self.position.size)  # Calculate units as an integer
            if units_to_sell > 0:
                self.sell(size=units_to_sell)
                # Reset stop-loss price since we've sold part of the position
                self.stop_price = None


class PostVolumeBasedStrategy(Strategy):

    def init(self):
        # Compute the 7-day rolling average of post volume (168 periods if data is hourly)
        self.rolling_avg_volume = self.I(
            lambda: pd.Series(self.data.post_volume).rolling(window=7 * 24).mean(),
            name='7-Day Rolling Average Post Volume'
        )

        # Compute the derivative (rate of change) of the rolling average volume
        self.rolling_avg_volume_derivative = self.I(
            lambda: pd.Series(self.rolling_avg_volume).diff(),
            name='Derivative of 7-Day Rolling Average Post Volume'
        )

    def next(self):
        # Check if the rolling average volume is below 1
        if self.rolling_avg_volume[-1] < 1:
            # Sell all holdings by specifying `size=self.position.size`
            self.sell(size=self.position.size)
            return  # Exit early to prevent further actions in this step

        # Set the trade size as the absolute value of the rolling average volume derivative
        scale = 5000
        units = int(abs(self.rolling_avg_volume_derivative[-1]) * scale)

        # Ensure units is at least 1 to avoid assertion errors
        if units == 0:
            return  # Skip trading if derivative is too small

        # Check if derivative goes from negative to positive (buy signal)
        if self.rolling_avg_volume_derivative[-2] < 0 and self.rolling_avg_volume_derivative[-1] > 0:
            # Buy a number of units equal to the derivative value
            self.buy(size=units)

        # Check if derivative goes from positive to negative (sell signal)
        elif self.rolling_avg_volume_derivative[-2] > 0 and self.rolling_avg_volume_derivative[-1] < 0:
            # Sell a number of units equal to the derivative value
            self.sell(size=units)


class SmaCrossVolume(Strategy):
    n1 = 10
    n2 = 20
    volume_threshold = 1.0  # Set a threshold for the rolling average volume

    def init(self):
        # Calculate the SMAs
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

        # Calculate the 7-day rolling average of post volume
        self.rolling_avg_volume = self.I(
            lambda: pd.Series(self.data.post_volume).rolling(window=7 * 24).mean(),
            name='7-Day Rolling Average Post Volume'
        )

    def next(self):
        # Check for buy signal (SMA crossover) with volume condition
        if crossover(self.sma1, self.sma2) and self.rolling_avg_volume[-1] > self.volume_threshold:
            self.buy()

        # Check for sell signal (SMA crossover) with volume condition
        elif crossover(self.sma2, self.sma1) and self.rolling_avg_volume[-1] < self.volume_threshold:
            self.sell()

# # Run backtest
# bt = Backtest(data, SmaCross, cash=10000, commission=.002, exclusive_orders=True)
# output = bt.run()
# bt.plot()
# #
# # # Run backtest
# bt = Backtest(data, MonthlyBuyHold, cash=10000, commission=.002, exclusive_orders=True)
# output = bt.run()
# bt.plot()
# #
# # Run backtest
# bt = Backtest(data, PostVolumeBasedStrategy, cash=10000, commission=.002, exclusive_orders=True)
# output = bt.run()
# bt.plot()


# Run backtest
bt = Backtest(data, SmaCrossVolume, cash=10000, commission=.002, exclusive_orders=True)
output = bt.run()
bt.plot()
