import pymongo
from pymongo import MongoClient
import os
import pandas as pd

# MongoDB connection setup
client = MongoClient(
    'mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.news
collection = db.newsweb_ulti

all_documents = collection.find()

df = pd.DataFrame(list(all_documents))
df['time'] = pd.to_datetime(df['time'])  # Ensure Date is in datetime format
df.set_index('time', inplace=True)  # Set Date as the index

df["content_cleaned"] = df["content"].str.extract(r"(?i)Share\s*message\s*(.*?)\s*", expand=False)

print(df.columns)
print(df.shape)
print(df.iloc[2])
# print(df.iloc[100]["content"])

