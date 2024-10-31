import pymongo
import pandas as pd
from datetime import datetime
from pymongo import MongoClient
import re
import os

# Set up MongoDB connection
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client["osebx"]

# Path to your directory
directory_path = "osebx_data/"

# Regular expression to match "OSL_DLY <stock>, <timeframe>.csv"
pattern = re.compile(r"OSL_DLY\s(.+?),\s(\d+[SMHD])\.csv")

# Loop through files in the directory, sorted alphabetically
for filename in sorted(os.listdir(directory_path)):
    # Check if filename matches the pattern
    match = pattern.search(filename)
    if match:
        stock = match.group(1)
        timeframe = match.group(2)

        # Define collection name as <stock>_<timeframe>
        collection_name = f"{stock}_{timeframe}"

        # Check if the collection already exists
        if collection_name in db.list_collection_names():
            print(f"Collection {collection_name} already exists. Skipping creation.")
            continue  # Skip to the next file if collection exists

        # Create as a time-series collection
        db.create_collection(collection_name, timeseries={"timeField": "time"})
        collection = db[collection_name]

        # Display collection creation confirmation
        print(f"Created time-series collection: {collection_name}")

        # Read the CSV file into a DataFrame
        file_path = os.path.join(directory_path, filename)
        data = pd.read_csv(file_path)

        # Convert 'time' from epoch to datetime if needed
        if data['time'].dtype == 'int64' or data['time'].dtype == 'float64':
            data['time'] = pd.to_datetime(data['time'], unit='s')

        # Convert DataFrame rows to dictionaries and insert into MongoDB
        documents = data.to_dict(orient="records")
        if documents:  # Ensure there is data to insert
            collection.insert_many(documents)
            print(f"Inserted {len(documents)} documents into {collection_name}")
        else:
