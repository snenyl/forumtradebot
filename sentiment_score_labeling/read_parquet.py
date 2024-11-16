import pandas as pd

# File path to the Parquet file
file_path = "combined_metrics_and_stock_data.parquet"

# Read the Parquet file
df = pd.read_parquet(file_path)

# Display basic information about the DataFrame
print("DataFrame Information:")
print(df.info())

# Display the first few rows of the DataFrame
print("\nFirst 5 Rows:")
print(df.head())

# Optional: Display summary statistics
print("\nSummary Statistics:")
print(df.describe())
