from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv
from config import MODEL_OPTIONS, INDEX_NAMES
from utils.embedding_utils import initialize_pinecone
from utils.search_utils import search_similar_items
from utils.related_search_utils import find_related_products  # Import from related_search_utils
import csv

# Initialize the Flask application
app = Flask(__name__)

# Load environment variables
load_dotenv()

result_count = 50  # Number of results to return

# Path to the CSV file
csv_file_path = 'event_log.csv'

# Ensure the CSV file exists and write header if it doesn't
if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Event Type', 'Search Query', 'Short Description', 'Product ID'])

# Initialize Pinecone
pc = initialize_pinecone(os.getenv('PINECONE_APIKEY'))

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

    index = pc.Index(index_name)
    search_results = search_similar_items(query, index, model_name, cutoff_percentage, result_count)

    print("Sending response back to client.")
    return jsonify({
        'search_results': search_results
    })

# Define the route for fetching related products
@app.route('/related-products', methods=['GET'])
def related_products():
    short_descr = request.args.get('short_descr')
    index_name = request.args.get('index_name')
    model_name = request.args.get('model_name')

    related_products = find_related_products(short_descr, index_name, model_name, pc)

    return jsonify({'related_products': related_products})

@app.route('/log-event', methods=['POST'])
def log_event():
    event_data = request.json

    # Extract data from the JSON
    event_type = event_data.get('event_type')
    product_info = event_data.get('product_info', {})
    search_query = event_data.get('search_query')
    short_descr = product_info.get('shortDescr', '')
    product_id = product_info.get('productId', '')

    # Append the event to the CSV file
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([event_type, search_query, short_descr, product_id])

    return jsonify({'status': 'success'})

# Start the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
