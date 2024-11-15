from pymongo import MongoClient

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')
db = client.tekinvestor
collection = db.pci_biotech_llama_10

# Creating the Python dictionary with descriptors and corresponding values
descriptors = {
    "sentiment": {
        "Extremely Negative": -1.0,
        "Very Negative": -0.8,
        "Negative": -0.6,
        "Slightly Negative": -0.4,
        "Neutral-Negative": -0.2,
        "Neutral": 0.0,
        "Neutral-Positive": 0.2,
        "Slightly Positive": 0.4,
        "Positive": 0.6,
        "Very Positive": 0.8,
        "Extremely Positive": 1.0,
    },
    "outlook": {
        "Utterly Hopeless": -1.0,
        "Deeply Pessimistic": -0.8,
        "Pessimistic": -0.6,
        "Cautiously Pessimistic": -0.4,
        "Skeptical": -0.2,
        "Neutral": 0.0,
        "Cautiously Optimistic": 0.2,
        "Hopeful": 0.4,
        "Optimistic": 0.6,
        "Very Optimistic": 0.8,
        "Extremely Optimistic": 1.0,
    },
    "credibility": {
        "Completely Unreliable": -1.0,
        "Highly Questionable": -0.8,
        "Unconvincing": -0.6,
        "Dubious": -0.4,
        "Somewhat Questionable": -0.2,
        "Neutral": 0.0,
        "Somewhat Trustworthy": 0.2,
        "Reliable": 0.4,
        "Trustworthy": 0.6,
        "Very Dependable": 0.8,
        "Extremely Credible": 1.0,
    },
    "referential_depth": {
        "Non-Existent": -1.0,
        "Superficial": -0.8,
        "Sparse": -0.6,
        "Minimal": -0.4,
        "Limited": -0.2,
        "Moderate": 0.0,
        "Detailed": 0.2,
        "Thorough": 0.4,
        "Comprehensive": 0.6,
        "Extensive": 0.8,
        "Exhaustive": 1.0,
    },
}

# Iterate over the documents in the collection
iterated = 0
# Iterate over the documents in the collection
for document in collection.find():
    updates = {}

    # Process each descriptor type
    for field, mapping in descriptors.items():
        # Get the corresponding numerical field name
        numerical_field = f"{field}_value"

        # Check if the numerical field already exists
        if numerical_field not in document:
            # Get the text descriptor value from the document
            descriptor = document.get(field)
            if descriptor in mapping:
                # Map the text descriptor to its numerical value
                updates[numerical_field] = mapping[descriptor]

    iterated += 1
    print(f"Iteration: {iterated}")

    # Update the document if there are changes
    if updates:
        collection.update_one({'_id': document['_id']}, {'$set': updates})

        print(f"Updated document ID {document['_id']} with: {updates}")
    else:
        print(f"Skipped document ID {document['_id']} (already up-to-date)")
