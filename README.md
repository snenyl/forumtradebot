# forumtradebot
AI-driven bot using TekInvestor forum sentiment &amp; historical backtesting for trading.


Gathering timeseries data from a specific stom from:
- Finansavisen
- Tekinvestor
- https://newsweb.oslobors.no/
- Tradingview / Stock data 




Forum data
- Post volume (Number of post in a given dataframe)

Sentiment score at thee different section post lengths 1 post, 20posts
and 100 posts. 



LSTM network:
3 outputs 1st close price, 5th close price 10th close price

30 Batch size
30 Sequence length
5 feature size (open, high, low, close, volume, rsi etc.. post volumne, max_post_length, min_post_length, avg_post_length etc..)

(30,30,5) 3 dimmentions.