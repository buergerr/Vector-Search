from flask import Flask, request, render_template, jsonify
import os
from transformers import AutoTokenizer, AutoModel
import torch
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from utils.model_utils import initialize_model_and_tokenizer, move_model_to_device
from config import MODEL_OPTIONS, INDEX_NAMES  # Import model and index options

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

result_count = 50  # Number of results to return

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_APIKEY'))

# Cache embeddings for frequently used queries
@lru_cache(maxsize=128)
def get_cached_embedding(query, tokenizer, model, device):
    print(f"Creating embedding for query: {query}")
    return create_embeddings([query], tokenizer, model, device)[0]

# Function to create embeddings
def create_embeddings(text_list, tokenizer, model, device):
    print(f"Creating embeddings for text list: {text_list}")
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
def search_similar_items(query, index_name, model_name, cutoff_percentage, top_k=result_count):
    print(f"Search started with model '{model_name}' and index '{index_name}'")

    tokenizer, model = initialize_model_and_tokenizer(model_name)
    device = move_model_to_device(model)

    print(f"Model and tokenizer initialized. Device: {device}")

    index = pc.Index(index_name)
    query_embedding = get_cached_embedding(query, tokenizer, model, device)

    print(f"Query embedding created: {query_embedding}")

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_values=True,  # Include the stored embeddings in the response
        include_metadata=True
    )

    print(f"Query to Pinecone index completed. Number of matches: {len(results['matches'])}")

    search_results = []
    base_url = os.getenv('PRODUCT_IMG_BASE_URL')

    for match in results['matches']:
        image_url = base_url + match['metadata'].get('imageurl', 'default_image.jpg') + os.getenv('PRODUCT_IMG_SUFFIX')
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

    print("Sorting search results by similarity score.")
    search_results = sorted(search_results, key=lambda x: x['similarity_score'], reverse=True)

    if search_results:
        highest_similarity_score = search_results[0]['similarity_score']
        cutoff_threshold = highest_similarity_score * (cutoff_percentage / 100.0)
        search_results = [result for result in search_results if result['similarity_score'] >= cutoff_threshold]

    relevances = [result['similarity_score'] for result in search_results]
    ndcg_value = ndcg(relevances)

    print(f"Search completed. NDCG value: {ndcg_value}")
    return search_results, ndcg_value

# Define the route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index_realtime.html', model_options=MODEL_OPTIONS, index_names=INDEX_NAMES)

# Define the route for real-time search
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    index_name = request.form['index_name']
    model_name = request.form['model_name']
    cutoff_percentage = float(request.form['cutoff'])

    print(f"Received search request. Query: '{query}', Index: '{index_name}', Model: '{model_name}', Cutoff: {cutoff_percentage}%")

    search_results, ndcg_value = search_similar_items(query, index_name, model_name, cutoff_percentage)

    print("Sending response back to client.")
    return jsonify({
        'search_results': search_results,
        'ndcg_value': ndcg_value
    })

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
