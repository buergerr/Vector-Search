import os
import numpy as np
from utils.embedding_utils import get_cached_embedding, normalize_text
from utils.search_utils import get_model_and_tokenizer

def find_related_products(short_descr, index_name, model_name, pc, top_k=5):
    print(f"Received request for related products for: {short_descr}")

    # Normalize the short description before embedding
    normalized_short_descr = normalize_text(short_descr)

    # Fetch the embedding for the selected short description
    tokenizer, model, device = get_model_and_tokenizer(model_name)
    product_embedding = get_cached_embedding(normalized_short_descr, tokenizer, model, device)

    # Query Pinecone for similar products using the stored embedding
    index = pc.Index(index_name)
    results = index.query(
        vector=product_embedding.tolist(),
        top_k=top_k,  # Number of related products to fetch
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

    return related_products
