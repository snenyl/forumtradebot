import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from sympy.physics.units import current

# Load main data and post volume data
data = pd.read_csv('data/OSL_DLY ULTI, 1H.csv', parse_dates=True, index_col='time')
data.index = pd.to_datetime(data.index, unit='s')
data = data[data.index >= '2019-02-25']

# Rename columns for backtesting
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(axis=1, how='all')

# Define strategy with post_volume as a custom indicator
class SmaCrossWithTrailingStop(Strategy):
    n1 = 10
    n2 = 20
    trailing_stop_percent = 0.02  # 2% trailing stop loss


    def init(self):
        close = self.data.Close
        self.sma1 = self.I(SMA, close, self.n1)
        self.sma2 = self.I(SMA, close, self.n2)

        # Track highest price after buying and lowest price after shorting
        self.entry_price = None
        self.trailing_stop_price = None

    def next(self):
        current_price = self.data.Close[-1]

        account_balance = self.equity
        risk_per_trade = 0.01  # Risking 1% of account balance
        max_risk_amount = account_balance * risk_per_trade

        # Calculate size only if trailing_stop_price is set and difference is non-zero
        if self.trailing_stop_price is not None and abs(current_price - self.trailing_stop_price) > 0:
            size = max_risk_amount / abs(current_price - self.trailing_stop_price)
        else:
            size = max_risk_amount / current_price  # Default size when no trailing stop or zero diff

        # Ensure size is a positive whole number or a fraction of equity
        size = max(1, int(size)) if size < float('inf') else 1  # Set to 1 if size becomes infinity


        # Check trailing stop loss first if a position is open
        if self.position:
            # Update trailing stop for a long position
            if self.position.is_long:
                if current_price > self.entry_price:
                    # Adjust trailing stop price
                    self.trailing_stop_price = max(self.trailing_stop_price,
                                                   current_price * (1 - self.trailing_stop_percent))

                # Check if the trailing stop loss price is hit
                if current_price <= self.trailing_stop_price:
                    self.position.close()
                    self.entry_price = None
                    self.trailing_stop_price = None
                    return  # Exit early if trailing stop triggered

        # Check for buy signal
        if crossover(self.sma1, self.sma2):
            self.buy(size=size)
            self.entry_price = current_price
            self.trailing_stop_price = current_price * (1 - self.trailing_stop_percent)

        # Check for sell signal
        elif crossover(self.sma2, self.sma1):
            self.sell(size=size)
            self.entry_price = None
            self.trailing_stop_price = None

# Run backtest
bt = Backtest(data, SmaCrossWithTrailingStop, cash=10000, commission=.002, exclusive_orders=True)
output = bt.run()
bt.plot()