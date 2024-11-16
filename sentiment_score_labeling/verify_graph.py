import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')
db_tekinvestor = client.tekinvestor
db_osebx = client.osebx
collection_pci_biotech_tek = db_tekinvestor.pci_biotech_llama_10
collection_pci_biotech_osebx = db_osebx.PCIB_1D

# Fetch data from MongoDB
data = list(collection_pci_biotech_tek.find({}, {
    "post_published_date": 1,
    "content": 1,
    "author": 1,
    "credibility_value": 1,
    "outlook_value": 1,
    "referential_depth_value": 1,
    "sentiment_value": 1,
    "_id": 0
}))

# Fetch data from MongoDB
stock_data = list(collection_pci_biotech_osebx.find({}, {
    "time": 1,
    "close": 1,
    "Volume": 1,
    "Histogram": 1,
    "Signal": 1,
    "MACD": 1,
    "_id": 0  # Optional, if you don't want the `_id` field
}))

# Convert to DataFrame
df = pd.DataFrame(data)
df_stock = pd.DataFrame(stock_data)

# Convert the 'time' field to datetime
df_stock['time'] = pd.to_datetime(df_stock['time'])
df_stock.set_index('time', inplace=True) # Set 'time' as the index
df_stock = df_stock.loc['2016-09-05':] # Filter the data to start from 2016-09-05

df_stock['close_pct_change'] = df_stock['close'].pct_change() * 100

# Normalize Volume and close using Min-Max normalization
df_stock['Volume'] = (df_stock['Volume'] - df_stock['Volume'].min()) / (df_stock['Volume'].max() - df_stock['Volume'].min())
df_stock['close_pct_change'] = 2 * ((df_stock['close'].pct_change() - df_stock['close'].pct_change().min()) /
                                    (df_stock['close'].pct_change().max() - df_stock['close'].pct_change().min())) - 1# df_stock['close'] = (df_stock['close'] - df_stock['close'].min()) / (df_stock['close'].max() - df_stock['close'].min())

# Calculate the 5-day percentage change in closing price
df_stock['close_pct_change_5d'] = df_stock['close'].pct_change(periods=5) * 100  # 5-day percentage change

# Get the maximum and minimum values of close_pct_change_5d
max_change_5d = df_stock['close_pct_change_5d'].max()  # Maximum percentage change
min_change_5d = df_stock['close_pct_change_5d'].min()  # Minimum percentage change

# Normalize close_pct_change_5d to range [-1, 1], preserving positive and negative signs
df_stock['close_pct_change_5d'] = df_stock['close_pct_change_5d'] / max(abs(max_change_5d), abs(min_change_5d))

# Add post_date column containing only the date part
df_stock['post_date'] = df_stock.index.date  # Extract the date from the datetime index
df_stock.set_index('post_date', inplace=True) # Set post_date as the new index
df_stock.index = pd.to_datetime(df_stock.index) # Ensure the index is a datetime object

# print(df_stock.head())
# exit(0)

# Ensure date parsing for plotting
df['post_published_date'] = pd.to_datetime(df['post_published_date'])

# Assuming 'content' column contains strings
df['total_words'] = df['content'].str.count(' ')+1

# Sort data by date
df = df.sort_values(by='post_published_date')

# Fill missing values (if any)
df = df.fillna(0)

# Create a daily average DataFrame
df['post_date'] = df['post_published_date'].dt.date  # Extract just the date part
daily_avg = df.groupby('post_date')[["credibility_value", "outlook_value", "referential_depth_value", "sentiment_value"]].mean().reset_index()

# Add the '12:00' timeframe to each date
daily_avg['post_published_date'] = pd.to_datetime(daily_avg['post_date']) + pd.Timedelta(hours=12)

# Add a daily count of samples
daily_counts = df.groupby('post_date').size().reset_index(name='daily_count')

# Maximum number of words for daily sample count
daily_max_words = df.groupby('post_date')['total_words'].max().reset_index()
daily_max_words.rename(columns={'total_words': 'max_words'}, inplace=True)

# Normalize max_words (scaling between 0 and 1)
max_words_min = daily_max_words['max_words'].min()
max_words_max = daily_max_words['max_words'].max()
daily_max_words['max_words_normalized'] = 2 * (daily_max_words['max_words'] - max_words_min) / (max_words_max - max_words_min) - 1

# Add 'post_published_date' for plotting
daily_max_words['post_published_date'] = pd.to_datetime(daily_max_words['post_date']) + pd.Timedelta(hours=12)



# Set indices for merging
daily_avg.set_index('post_date', inplace=True)
daily_counts.set_index(pd.to_datetime(daily_counts['post_date']), inplace=True)
daily_max_words.set_index('post_date', inplace=True)

# print(daily_avg.head())  # Inspect the daily_avg DataFrame
# print(daily_counts.head())  # Inspect the daily_counts DataFrame
# print(daily_max_words.head())  # Inspect the daily_max_words DataFrame
# print(df_stock.head())  # Inspect the daily_max_words DataFrame
# exit(0)

# Z-score normalization for daily sample count
mean_count = daily_counts['daily_count'].mean()
std_count = daily_counts['daily_count'].std()
daily_counts['z_score'] = (daily_counts['daily_count'] - mean_count) / std_count

# Scale Z-scores between -1 and 1
z_min = daily_counts['z_score'].min()
z_max = daily_counts['z_score'].max()
daily_counts['z_score_scaled'] = 2 * (daily_counts['z_score'] - z_min) / (z_max - z_min) - 1


#Snoeffelen start

snoeffelen_df = df[df['author'] == 'Snoeffelen'] # Filter data for the specific author
snoeffelen_df['post_published_date'] = pd.to_datetime(snoeffelen_df['post_published_date']) # Ensure date parsing for plotting
snoeffelen_df = snoeffelen_df.sort_values(by='post_published_date') # Sort data by date
snoeffelen_df = snoeffelen_df.fillna(0) # Fill missing values (if any)
snoeffelen_daily_samples = snoeffelen_df.groupby(snoeffelen_df['post_published_date'].dt.date).size().reset_index(name='daily_sample_count') # Count daily samples (posts) for Snoeffelen
snoeffelen_daily_samples.rename(columns={"post_published_date": "date"}, inplace=True) # Rename columns for clarity

# Normalize the daily sample count
min_samples = snoeffelen_daily_samples['daily_sample_count'].min() # Find the minimum value
max_samples = snoeffelen_daily_samples['daily_sample_count'].max() # Find the maximum value
snoeffelen_daily_samples['normalized_daily_sample_count'] =2 * (snoeffelen_daily_samples['daily_sample_count'] - min_samples) / (max_samples - min_samples) - 1# Apply Min-Max normalization


#Snoeefelen end test



# Create a scatter plot for raw values and a line plot for daily averages
fig = make_subplots(specs=[[{"secondary_y": True}]])


# Add the financial data to the plot
fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['Volume'],
        mode='lines',
        name="Volume",
        line=dict(width=2, dash="dot")
    ),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['close_pct_change'],
        mode='lines',
        name="Close Procent Change",
        line=dict(width=2)
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['close_pct_change_5d'],
        mode='lines',
        name="Close Percent Change 5D",
        line=dict(width=2)
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['Histogram'],
        mode='lines',
        name="Histogram",
        line=dict(width=2)
    ),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['Signal'],
        mode='lines',
        name="Signal",
        line=dict(width=2, dash="dot")
    ),
    secondary_y=True
)

fig.add_trace(
    go.Scatter(
        x=df_stock.index,
        y=df_stock['MACD'],
        mode='lines',
        name="MACD",
        line=dict(width=2, color="green")
    ),
    secondary_y=True
)


metrics = ["credibility_value", "outlook_value", "referential_depth_value", "sentiment_value"]

# Add scatter plots for raw values
# for metric in metrics:
#     fig.add_trace(
#         go.Scatter(
#             x=df['post_published_date'],
#             y=df[metric],
#             mode='markers',
#             name=f"{metric} (Raw)",
#             marker=dict(size=4)
#         )
#     )

# Add line plots for daily averages
for metric in metrics:
    fig.add_trace(
        go.Scatter(
            x=daily_avg['post_published_date'],
            y=daily_avg[metric],
            mode='lines',
            name=f"{metric} (Daily Avg)",
            line=dict(width=2)
        )
    )

# Add line plot for scaled Z-score daily sample counts
fig.add_trace(
    go.Scatter(
        x=daily_counts['post_date'],
        y=daily_counts['z_score_scaled'],
        mode='lines',
        name="Z-Score Scaled Daily Sample Count",
        line=dict(width=2),
        yaxis="y2"
    )
)


# Add a line plot for normalized max words
fig.add_trace(
    go.Scatter(
        x=daily_max_words['post_published_date'],
        y=daily_max_words['max_words_normalized'],
        mode='lines',
        name="Normalized Daily Max Words",
        line=dict(width=2)  # Dotted line for distinct visualization
    )
)

# Update layout with secondary y-axis for scaled Z-score daily sample count
fig.update_layout(
    title="Metrics Over Time with Z-Score Scaled Daily Sample Count",
    xaxis_title="Post Published Date",
    yaxis_title="Value",
    legend_title="Metrics",
    template="plotly_white",
    yaxis2=dict(
        title="Z-Score Scaled Daily Sample Count",
        overlaying='y',
        side='right',
        showgrid=False
    )
)

# Snoeffelen
fig.add_trace(
    go.Scatter(
        x=snoeffelen_daily_samples['date'], # Use the date column for the x-axis
        y=snoeffelen_daily_samples['normalized_daily_sample_count'], # Use the normalized daily sample count for the y-axis
        mode='lines', # Show both lines and markers
        name="Snoeffelen Normalized Daily Sample Count", # Legend entry name
        line=dict(width=2, color="blue") # Customize line width and color
    )
)

# Save the plot as an HTML file
output_html_path = "metrics_with_z_score_scaled_counts_plot.html"
fig.write_html(output_html_path)
output_html_path

#Save all the data that is used for the graphs as parquet in a single dataframe with the same time index
merged_df = df_stock.copy()

for metric in ["credibility_value", "outlook_value", "referential_depth_value", "sentiment_value"]:
    merged_df[metric] = daily_avg[metric]

merged_df['z_score_scaled'] = daily_counts['z_score_scaled']
merged_df['max_words_normalized'] = daily_max_words['max_words_normalized']

# Fill missing values
merged_df.fillna(0, inplace=True)

# Save the merged DataFrame as a Parquet file
merged_df.to_parquet("combined_metrics_and_stock_data.parquet")
print("Data saved to 'combined_metrics_and_stock_data.parquet'")