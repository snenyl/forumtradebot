import pymongo
from pymongo import MongoClient
import os

# MongoDB connection setup
client = MongoClient(
    'mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.tekinvestor
collection = db.pci_biotech

