import pymongo
import pandas as pd
from datetime import datetime
from pymongo import MongoClient

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.tekinvestor
collection = db.ultimovacs

# Fetch all documents and convert to a DataFrame
posts = list(collection.find({}, {"post_published_date": 1}))
data = pd.DataFrame(posts)

# Convert MongoDB date format to datetime and set up the DataFrame
data['post_published_date'] = pd.to_datetime(data['post_published_date'])
data.set_index('post_published_date', inplace=True)

# Resample data by 1 hour and count posts
post_volume = data.resample('1H').size()

# Convert to DataFrame and reset index
post_volume_df = post_volume.reset_index(name='post_volume')

# Save to CSV
output_file = "data/post_volume_by_hour_ulti.csv"
post_volume_df.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")



import plotly.express as px
# Load the data
file_path = "data/post_volume_by_hour.csv"  # Adjusted for execution environment
post_volume_df = pd.read_csv(file_path)
# Plot using Plotly
fig = px.line(post_volume_df, x='post_published_date', y='post_volume', title='Post Volume by Hour',

              labels={'post_published_date': 'Published Date', 'post_volume': 'Post Volume'})
fig.show()


