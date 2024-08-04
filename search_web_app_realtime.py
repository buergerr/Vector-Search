from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from config import MODEL_OPTIONS, INDEX_NAMES
from utils.embedding_utils import create_embeddings, get_cached_embedding, compare_embeddings, initialize_pinecone, normalize_text
from utils.model_utils import initialize_model_and_tokenizer, move_model_to_device
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables from .env file
load_dotenv()

result_count = 50  # Number of results to return

# Initialize Pinecone
pc = initialize_pinecone(os.getenv('PINECONE_APIKEY'))

# Global dictionary to store the model and tokenizer
model_store = {}

# Function to load or get model and tokenizer
def get_model_and_tokenizer(model_name):
    if model_name not in model_store:
        print(f"Initializing model and tokenizer for: {model_name}")
        tokenizer, model = initialize_model_and_tokenizer(model_name)
        device = move_model_to_device(model)  # Move model to device and capture the device
        model_store[model_name] = (tokenizer, model, device)
    else:
        print(f"Reusing model and tokenizer for: {model_name}")
    return model_store[model_name]

# Function to search for similar items
def search_similar_items(query, index_name, model_name, cutoff_percentage, top_k=result_count):
    print(f"Search started with model '{model_name}' and index '{index_name}'")

    # Normalize the search query
    normalized_query = normalize_text(query)

    tokenizer, model, device = get_model_and_tokenizer(model_name)
    
    index = pc.Index(index_name)
    query_embedding = get_cached_embedding(normalized_query, tokenizer, model, device)

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
        stored_embedding = np.array(match['values'])

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


@app.route('/health', methods=['GET'])
def health():
    resp = jsonify(health="healthy")
    resp.status_code = 200

    return resp

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

@app.route('/related-products', methods=['GET'])
def related_products():
    short_descr = request.args.get('short_descr')
    print(f"Received request for related products for: {short_descr}")

    # Normalize the short description before embedding
    normalized_short_descr = normalize_text(short_descr)

    index_name = request.args.get('index_name')  # Get index_name from query string

    # Fetch the embedding for the selected short description
    tokenizer, model, device = get_model_and_tokenizer(request.args.get('model_name'))
    product_embedding = get_cached_embedding(normalized_short_descr, tokenizer, model, device)

    # Query Pinecone for similar products using the stored embedding
    index = pc.Index(index_name)
    results = index.query(
        vector=product_embedding.tolist(),
        top_k=10,  # Number of related products to fetch
        include_values=True,
        include_metadata=True
    )

    related_products = []
    base_url = os.getenv('PRODUCT_IMG_BASE_URL')
    suffix = os.getenv('PRODUCT_IMG_SUFFIX')

    for match in results['matches']:
        if match['metadata']['shortdescrdisplay'] != short_descr:  # Exclude the clicked product from related products
            image_url = base_url + match['metadata'].get('imageurl', 'default_image.jpg') + suffix
            related_products.append({
                'image_url': image_url,
                'short_descr': match['metadata'].get('shortdescrdisplay', 'No description available'),
                'minprice': match['metadata'].get('minprice', 'N/A'),
                'manufacturer': match['metadata'].get('manufacturer', 'Unknown')
            })

    return jsonify({'related_products': related_products})


# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
