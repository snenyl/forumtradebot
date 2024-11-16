import pandas as pd

# Path to your .parquet file
file_path = "data/r876_v5_0_train.parquet"

# Read the .parquet file into a pandas DataFrame
df = pd.read_parquet(file_path)

# # Filter rows where id is 'n0007b5abb0c3a25'
# filtered_df = df[df['id'] == 'n0007b5abb0c3a25']
#
# # Display the filtered rows
# print(filtered_df)

# Display the first few rows of the DataFrame
# Set pandas display options to show all columns
# pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.width', None)       # Adjust the width to fit the screen
print(df.head())
print(df.columns)
