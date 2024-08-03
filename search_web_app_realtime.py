from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from config import MODEL_OPTIONS, INDEX_NAMES
from utils.embedding_utils import create_embeddings, get_cached_embedding, compare_embeddings, initialize_pinecone
from utils.model_utils import initialize_model_and_tokenizer, move_model_to_device
import numpy as np  # Make sure to import numpy

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

result_count = 50  # Number of results to return

# Initialize Pinecone
pc = initialize_pinecone(os.getenv('PINECONE_APIKEY'))

# Function to search for similar items
def search_similar_items(query, index_name, model_name, cutoff_percentage, top_k=result_count):
    print(f"Search started with model '{model_name}' and index '{index_name}'")

    tokenizer, model = initialize_model_and_tokenizer(model_name)
    device = move_model_to_device(model)

    print(f"Model and tokenizer initialized. Device: {device}")

    index = pc.Index(index_name)
    query_embedding = get_cached_embedding(query, tokenizer, model, device)

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_values=True,
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
        stored_embedding = np.array(match['values'])  # Ensure numpy is imported

        # Calculate similarity between the query embedding and the stored embedding
        similarity_score = compare_embeddings(query_embedding, stored_embedding)

        search_results.append({
            'image_url': image_url,
            'short_descr': short_descr,
            'minprice': minprice,
            'manufacturer': manufacturer,
            'score': match['score'],
            'similarity_score': similarity_score
        })

    print("Sorting search results by similarity score.")
    search_results = sorted(search_results, key=lambda x: x['similarity_score'], reverse=True)

    if search_results:
        highest_similarity_score = search_results[0]['similarity_score']
        cutoff_threshold = highest_similarity_score * (cutoff_percentage / 100.0)
        search_results = [result for result in search_results if result['similarity_score'] >= cutoff_threshold]

    print("Search completed.")
    return search_results

# Define the route for the home page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', model_options=MODEL_OPTIONS, index_names=INDEX_NAMES)

# Define the route for real-time search
@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    index_name = request.form['index_name']
    model_name = request.form['model_name']
    cutoff_percentage = float(request.form['cutoff'])

    print(f"Received search request. Query: '{query}', Index: '{index_name}', Model: '{model_name}', Cutoff: {cutoff_percentage}%")

    search_results = search_similar_items(query, index_name, model_name, cutoff_percentage)

    print("Sending response back to client.")
    return jsonify({
        'search_results': search_results
    })

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
