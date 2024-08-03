# Import required libraries
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
import torch
import pinecone
from dotenv import load_dotenv
from pinecone import Pinecone
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import os
# Load environment variables from .env file
load_dotenv()


# Define the search query
query = "Strandkleid für den Sommer"
# Define the number of results to return
result_count = 500

print("Search query:", query)


# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')


# Move model to appropriate device (CPU or GPU)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

pc = Pinecone(api_key=os.getenv('PINECONE_APIKEY'))

index_name = 'product-embeddings' #change this to the name of the index you created
index = pc.Index(index_name, host=os.getenv('PINECONE_INDEX_HOST'))

# Function to create embeddings
def create_embeddings(text_list):
    inputs = tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Function to search for similar items
def search_similar_items(query, top_k=result_count):
    query_embedding = create_embeddings([query])[0]
    # Query the Pinecone index, ensuring values are included
    results = index.query(
        vector=query_embedding.tolist(), 
        top_k=top_k, 
        include_values=True,  # Ensure embeddings are included in the results
        include_metadata=True
    )
    return query_embedding, results


# Example search query

query_embedding, results = search_similar_items(query)

# Display the search results and inspect embeddings
print("Search results:")
for match in results['matches']:
    print(f"Short Description: {match['metadata']['shortdescrdisplay']}, Score: {match['score']}")
    #print(f"Embedding shape: {np.array(match['values']).shape}, Embedding: {match['values']}")

# Prepare the data for visualization
embs = [query_embedding]
metadata = ["Query: " + query]

# Check if the returned Pinecone embeddings have the same shape as query_embedding and are not empty
for match in results['matches']:
    emb_values = np.array(match['values'])
    if emb_values.shape == query_embedding.shape and emb_values.size > 0:
        embs.append(emb_values)
        metadata.append(f"Short Description: {match['metadata']['shortdescrdisplay']}")
    else:
        print(f"Skipping embedding with different shape or empty: {emb_values.shape}")

# Check if there are enough valid samples for t-SNE
if len(embs) > 1:
    # Convert to numpy array
    embs = np.array(embs)
    
    # Adjust perplexity to be less than the number of samples
    perplexity_value = min(5, len(embs) - 1)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity_value)
    reduced_vectors = tsne.fit_transform(embs)

    # Create a 3D scatter plot
    scatter_plot = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color='grey', opacity=0.5, line=dict(color='lightgray', width=1)),
        text=[metadata[i] for i in range(len(reduced_vectors))]
    )

    # Highlight the first point with a different color
    highlighted_point = go.Scatter3d(
        x=[reduced_vectors[0, 0]],
        y=[reduced_vectors[0, 1]],
        z=[reduced_vectors[0, 2]],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.8, line=dict(color='lightgray', width=1)),
        text=[metadata[0]]
    )

    # Highlight the top 3 points
    top_points = go.Scatter3d(
        x=reduced_vectors[1:4, 0],
        y=reduced_vectors[1:4, 1],
        z=[reduced_vectors[1:4, 2]],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.8, line=dict(color='black', width=1)),
        text=[metadata[i+1] for i in range(3)]
    )

    # Create the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title='3D Representation after t-SNE (Perplexity adjusted)'
    )

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Add the scatter plots to the figure
    fig.add_trace(scatter_plot)
    fig.add_trace(highlighted_point)
    fig.add_trace(top_points)

    fig.update_layout(layout)

    pio.write_html(fig, 'interactive_plot.html')
    fig.show()
else:
    print("Not enough valid samples for t-SNE visualization.")

