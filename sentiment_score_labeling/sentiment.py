import pymongo
import requests
import json
from pymongo import MongoClient

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.tekinvestor
collection = db.pci_biotech_llama

# Prompt prefix
prompt_prefix = """
Based on the content is the poster bullish, neutral or bearish? Give the output as a JSON ONLY in the following format:
{
"sentiment": ,
}
"""

# Initialize iteration counter
iteration = 0

# Fetch documents one at a time and update with sentiment if not already set
for post in collection.find({}, {"content": 1, "likes": 1, "post_published_date": 1, "sentiment": 1}):
    iteration += 1
    print(f"Processing post {iteration}, Post ID: {post['_id']}")

    # Check if 'sentiment' already exists; if so, skip this document
    if 'sentiment' in post and post['sentiment'] is not None:
        print(f"Sentiment already exists for Post ID {post['_id']}, skipping...")
        continue

    content = post.get("content", "")
    sentiment = None  # Default sentiment in case of failure

    if content:
        # Create the complete prompt
        prompt = prompt_prefix + content

        # Prepare the payload for the POST request
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }

        # Send the POST request
        try:
            response = requests.post("http://10.0.42.220:7869/api/generate", json=payload)
            json_response = json.loads(response.text)
            model_response = json.loads(json_response["response"])

            # Extract sentiment
            sentiment = model_response.get("sentiment")
            print(f"Post ID: {post['_id']}, Sentiment: {sentiment}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for post ID: {post['_id']} with error: {e}")
        except json.JSONDecodeError:
            print(f"Failed to parse response for post ID: {post['_id']}, Response: {response.text}")

    # Update the document in MongoDB with the new "sentiment" field
    collection.update_one({'_id': post['_id']}, {'$set': {'sentiment': sentiment}})
    print(f"Updated Post ID {post['_id']} with sentiment: {sentiment}")


# post_volume (number of posts in the giver 1H timeframe)
# max_content_length
# min_content_length
# average_post_length
# num_bullish (num or percentage of total within timeframe)
# num_neutral (num or percentage of total within timeframe)
# num_bearish (num or percentage of total within timeframe)
# unique_posters (number of different authors in the given timeframe)
# max_likes
# min_likes
# average_likes
# like_rate average_likes relative to post_volume

# # Convert MongoDB date format to datetime and set up the DataFrame
# data['post_published_date'] = pd.to_datetime(data['post_published_date'])
# data.set_index('post_published_date', inplace=True)
#
# # Resample data by 1 hour and count posts
# post_volume = data.resample('1H').size()
#
# # Convert to DataFrame and reset index
# post_volume_df = post_volume.reset_index(name='post_volume')
#
# # Save to CSV
# output_file = "data/post_volume_by_hour_ulti.csv"
# post_volume_df.to_csv(output_file, index=False)
#
# print(f"Data saved to {output_file}")



# import plotly.express as px
# # Load the data
# file_path = "data/post_volume_by_hour.csv"  # Adjusted for execution environment
# post_volume_df = pd.read_csv(file_path)
# # Plot using Plotly
# fig = px.line(post_volume_df, x='post_published_date', y='post_volume', title='Post Volume by Hour',
#
#               labels={'post_published_date': 'Published Date', 'post_volume': 'Post Volume'})
# fig.show()
#

