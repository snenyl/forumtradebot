import pymongo
import requests
import json
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# MongoDB connection setup
client = MongoClient('mongodb://mongoadmin:secret@10.0.0.150:27017/?authMechanism=DEFAULT')  # Adjust with your MongoDB URI if needed
db = client.tekinvestor
# collection = db.pci_biotech_llama_10
collection = db.photocure_llama_10
# collection = db.ultimovacs_llama_10

# Prompt prefix
prompt_prefix = """
Given the following post \n
"""


prompt_suffix = """
\n
Fill out the following JSON: 
Sentiment refers to the overall emotional tone or attitude conveyed in a post, reflecting the author’s subjective feelings such as positivity, negativity, or neutrality toward a subject, idea, or event with descriptors ranging from Extremely Negative to Extremely Positive.
Outlook represents the degree of optimism or pessimism regarding future events, developments, or possibilities, with descriptors ranging from Utterly Hopeless to Extremely Optimistic.
Credibility measures the perceived reliability, trustworthiness, and expertise of the author or content, ranging from Completely Unreliable to Extremely Credible.
Referential Depth evaluates the richness and diversity of references, ranging from Non-Existent to Exhaustive.  

The output should be formatted as a JSON OBJECT ONLY! in the following structure: 
{ "sentiment": "<descriptor: one of 'Extremely Negative', 'Very Negative', 'Negative', 'Slightly Negative', 'Neutral-Negative', 'Neutral', 'Neutral-Positive', 'Slightly Positive', 'Positive', 'Very Positive', 'Extremely Positive'>", "outlook": "<descriptor: one of 'Utterly Hopeless', 'Deeply Pessimistic', 'Pessimistic', 'Cautiously Pessimistic', 'Skeptical', 'Neutral', 'Cautiously Optimistic', 'Hopeful', 'Optimistic', 'Very Optimistic', 'Extremely Optimistic'>", "credibility": "<descriptor: one of 'Completely Unreliable', 'Highly Questionable', 'Unconvincing', 'Dubious', 'Somewhat Questionable', 'Neutral', 'Somewhat Trustworthy', 'Reliable', 'Trustworthy', 'Very Dependable', 'Extremely Credible'>", "referential_depth": "<descriptor: one of 'Non-Existent', 'Superficial', 'Sparse', 'Minimal', 'Limited', 'Moderate', 'Detailed', 'Thorough', 'Comprehensive', 'Extensive', 'Exhaustive'>" } 

"""

# Shared counter and lock for thread-safe updates
processed_count = 0
count_lock = Lock()

# Function to process a single post
def process_post(post):
    global processed_count
    post_id = post['_id']
    content = post.get("content", "")
    sentiment = outlook = credibility = referential_depth = None  # Default values in case of failure

    # Check if all fields already exist; if so, skip this document
    if all(key in post and post[key] is not None for key in ['sentiment', 'outlook', 'credibility', 'referential_depth']):
        print(f"Descriptors already exist for Post ID {post_id}, skipping...")

        with count_lock:
            processed_count += 1
            print(f"Total posts processed: {processed_count}")
        return None

    if content:
        # Create the complete prompt
        prompt = prompt_prefix + content + prompt_suffix

        # Prepare the payload for the POST request
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }

        # Send the POST request
        try:
            response = requests.post("http://10.0.42.220:7869/api/generate", json=payload)
            response.raise_for_status()
            json_response = json.loads(response.text)
            model_response = json.loads(json_response["response"])

            # Extract descriptors
            sentiment = model_response.get("sentiment")
            outlook = model_response.get("outlook")
            credibility = model_response.get("credibility")
            referential_depth = model_response.get("referential_depth")
            print(f"Post ID: {post_id}, Sentiment: {sentiment}, Outlook: {outlook}, Credibility: {credibility}, Referential Depth: {referential_depth}")

        except requests.exceptions.RequestException as e:
            print(f"Request failed for post ID: {post_id} with error: {e}")
        except json.JSONDecodeError:
            print(f"Failed to parse response for post ID: {post_id}, Response: {response.text}")

    # Update the document in MongoDB with the new fields
    update_data = {
        'sentiment': sentiment,
        'outlook': outlook,
        'credibility': credibility,
        'referential_depth': referential_depth
    }
    collection.update_one({'_id': post_id}, {'$set': update_data})
    print(f"Updated Post ID {post_id} with sentiment: {sentiment}, outlook: {outlook}, credibility: {credibility}, referential_depth: {referential_depth}")

    # Increment the shared counter in a thread-safe way
    with count_lock:
        processed_count += 1
        print(f"Total posts processed: {processed_count}")

# Main logic for using ThreadPoolExecutor
def main():
    posts = list(collection.find({}, {"content": 1, "likes": 1, "post_published_date": 1, "sentiment": 1, "outlook": 1, "credibility": 1, "referential_depth": 1}))
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_post, post) for post in posts]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in worker: {e}")

if __name__ == "__main__":
    main()



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

