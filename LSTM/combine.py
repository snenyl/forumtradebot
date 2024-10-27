import pandas as pd

# Load the datasets
df_stock = pd.read_csv('data/OSL_DLY PCIB, 1H.csv')
df_sentiment = pd.read_csv('data/metrics_output_pcib.csv')

# Convert the epoch timestamp in 'df_stock' to datetime
df_stock['time'] = pd.to_datetime(df_stock['time'], unit='s')

# Convert 'post_published_date' in 'df_sentiment' to datetime
df_sentiment['post_published_date'] = pd.to_datetime(df_sentiment['post_published_date'])

# Rename columns for merging to avoid confusion during the merge
df_stock.rename(columns={'time': 'timestamp'}, inplace=True)
df_sentiment.rename(columns={'post_published_date': 'timestamp'}, inplace=True)

# Perform the merge with an outer join to include all rows and columns
df_merged = pd.merge(df_stock, df_sentiment, on='timestamp', how='outer')

# Sort the merged DataFrame by timestamp for ordered output
df_merged.sort_values(by='timestamp', inplace=True)

# Reset the index after sorting
df_merged.reset_index(drop=True, inplace=True)

# Save the resulting DataFrame to a CSV file without dropping any empty columns
df_merged.to_csv('data/merged_output_pcib.csv', index=False)

# Display the head of the merged DataFrame for verification
print(df_merged.head())
