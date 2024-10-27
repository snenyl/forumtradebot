import pymongo
import requests
import json
from pymongo import MongoClient
import pandas as pd
import plotly.express as px

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.tekinvestor
collection = db.pci_biotech_llama

# Retrieve data from MongoDB
documents = list(collection.find())

# Convert MongoDB-like documents to a DataFrame
data = []
for doc in documents:
    data.append({
        "post_published_date": doc["post_published_date"],  # Already a datetime object
        "post_modified_date": doc["post_modified_date"],
        "forum_thead_id": doc["forum_thead_id"],
        "likes": doc.get("likes", 0),
        "content": doc.get("content", ""),
        "post_position": doc["post_position"],
        "forum_thread_last_modified": doc["forum_thread_last_modified"],
        "author": doc["author"],
        "sentiment": doc.get("sentiment", "neutral")
    })
df = pd.DataFrame(data)

# Set the index to 'post_published_date'
df.set_index('post_published_date', inplace=True)

# Calculate content length for each post
df['content_length'] = df['content'].str.len()

# Resample data by 1 hour
resampled = df.resample('1H')

# Calculate each metric
metrics = pd.DataFrame({
    "post_volume": resampled.size(),
    "max_content_length": resampled['content_length'].max(),
    "min_content_length": resampled['content_length'].min(),
    "average_post_length": resampled['content_length'].mean(),
    "percent_bullish": resampled['sentiment'].apply(lambda x: (x == 'bullish').sum() / len(x)),
    "percent_neutral": resampled['sentiment'].apply(lambda x: (x == 'neutral').sum() / len(x)),
    "percent_bearish": resampled['sentiment'].apply(lambda x: (x == 'bearish').sum() / len(x)),
    "unique_posters": resampled['author'].nunique(),
    "max_likes": resampled['likes'].max(),
    "min_likes": resampled['likes'].min(),
    "average_likes": resampled['likes'].mean(),
    "like_rate": resampled['likes'].mean() / resampled.size()
}).reset_index()  # Reset index for plotly

# save to csv
metrics.to_csv("metrics_output.csv", index=False)

# Plot each metric using Plotly Express
figs = []
metrics_to_plot = [
    "post_volume", "max_content_length", "min_content_length", "average_post_length",
    "percent_bullish", "percent_neutral", "percent_bearish", "unique_posters",
    "max_likes", "min_likes", "average_likes", "like_rate"
]

for metric in metrics_to_plot:
    fig = px.line(metrics, x='post_published_date', y=metric, title=f'{metric} over Time')
    fig.show()
    figs.append(fig)  # Store figures if you want to use them later


# post_volume (number of posts in the giver 1H timeframe)
# max_content_length
# min_content_length
# average_post_length
# num_bullish (num or percentage of total within timeframe)
# num_neutral (num or percentage of total within timeframe)
# num_bearish (num or percentage of total within timeframe)
# unique_posters (number of different authors in the given timeframe)
# max_likes
# min_likes
# average_likes
# like_rate average_likes relative to post_volume
