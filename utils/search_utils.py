import os
import numpy as np
from utils.embedding_utils import get_cached_embedding, compare_embeddings, normalize_text
from utils.model_utils import initialize_model_and_tokenizer, move_model_to_device

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
def search_similar_items(query, index, model_name, cutoff_percentage, top_k):
    print(f"Search started with model '{model_name}' and index '{index}'")

    # Normalize the search query
    normalized_query = normalize_text(query)

    tokenizer, model, device = get_model_and_tokenizer(model_name)
    
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
