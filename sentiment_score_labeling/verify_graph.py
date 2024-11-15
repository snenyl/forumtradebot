import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')
db = client.tekinvestor
collection = db.pci_biotech_llama_10

# Fetch data from MongoDB
data = list(collection.find({}, {
    "post_published_date": 1,
    "credibility_value": 1,
    "outlook_value": 1,
    "referential_depth_value": 1,
    "sentiment_value": 1,
    "_id": 0
}))

# Convert to DataFrame
df = pd.DataFrame(data)

# Ensure date parsing for plotting
df['post_published_date'] = pd.to_datetime(df['post_published_date'])

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

# Z-score normalization for daily sample count
mean_count = daily_counts['daily_count'].mean()
std_count = daily_counts['daily_count'].std()
daily_counts['z_score'] = (daily_counts['daily_count'] - mean_count) / std_count

# Scale Z-scores between -1 and 1
z_min = daily_counts['z_score'].min()
z_max = daily_counts['z_score'].max()
daily_counts['z_score_scaled'] = 2 * (daily_counts['z_score'] - z_min) / (z_max - z_min) - 1

# Create a scatter plot for raw values and a line plot for daily averages
fig = make_subplots(specs=[[{"secondary_y": True}]])

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

# Save the plot as an HTML file
output_html_path = "metrics_with_z_score_scaled_counts_plot.html"
fig.write_html(output_html_path)
output_html_path
