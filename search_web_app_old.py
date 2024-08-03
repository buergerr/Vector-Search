from flask import Flask, request, render_template
import os
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from model_utils import initialize_model_and_tokenizer, move_model_to_device, model_options  # Import model options

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

# Choose the model you want to use (e.g., the first one)
selected_model = model_options[5]
result_count = 50  # Number of results to return

# Initialize the selected model and tokenizer
tokenizer, model = initialize_model_and_tokenizer(selected_model)

# Move model to GPU if available, else use CPU
device = move_model_to_device(model)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_APIKEY'))

# Function to create embeddings
def create_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Function to compare embeddings using cosine similarity
def compare_embeddings(query_embedding, result_embedding):
    similarity = cosine_similarity([query_embedding], [result_embedding])
    return similarity[0][0]


# Function to calculate DCG
def dcg(relevances):
    return sum([rel / math.log2(idx + 2) for idx, rel in enumerate(relevances)])

# Function to calculate NDCG
def ndcg(relevances):
    dcg_value = dcg(relevances)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg_value = dcg(ideal_relevances)
    return dcg_value / idcg_value if idcg_value > 0 else 0

# Function to search for similar items and calculate NDCG
def search_similar_items(query, index_name, cutoff_percentage, top_k=result_count):
    index = pc.Index(index_name)
    query_embedding = create_embeddings([query])[0]
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_values=True,  # Include the stored embeddings in the response
        include_metadata=True
    )

    search_results = []
    base_url = os.getenv('PRODUCT_IMG_BASE_URL')

    for match in results['matches']:
        image_url = base_url + match['metadata'].get('imageurl', 'default_image.jpg')
        short_descr = match['metadata'].get('shortdescrdisplay', 'No description available')
        minprice = match['metadata'].get('minprice', 'N/A')
        manufacturer = match['metadata'].get('manufacturer', 'Unknown')

        # Use the stored embedding from Pinecone
        stored_embedding = np.array(match['values'])

        # Calculate similarity between the query embedding and the stored embedding
        similarity_score = compare_embeddings(query_embedding, stored_embedding)

        search_results.append({
            'image_url': image_url,
            'short_descr': short_descr,
            'minprice': minprice,
            'manufacturer': manufacturer,
            'score': match['score'],
            'similarity_score': similarity_score  # Add similarity score to the results
        })

    # Sort results by similarity score in descending order
    search_results = sorted(search_results, key=lambda x: x['similarity_score'], reverse=True)

    # Apply cutoff: Keep results within the given percentage of the highest similarity score
    if search_results:
        highest_similarity_score = search_results[0]['similarity_score']
        cutoff_threshold = highest_similarity_score * (cutoff_percentage / 100.0)
        search_results = [result for result in search_results if result['similarity_score'] >= cutoff_threshold]

    # Extract relevance scores (for demonstration, using the similarity scores as relevance)
    relevances = [result['similarity_score'] for result in search_results]

    # Calculate NDCG
    ndcg_value = ndcg(relevances)

    return search_results, ndcg_value


# Define the route for the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form['query']
        index_name = request.form['index_name']  # Get the selected index name from the form
        cutoff_percentage = float(request.form['cutoff'])  # Get the cutoff percentage from the form
        search_results, ndcg_value = search_similar_items(query, index_name, cutoff_percentage)

        return render_template('results.html', query=query, search_results=search_results, ndcg_value=ndcg_value, cutoff=cutoff_percentage, index_name=index_name)

    return render_template('index.html')

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
