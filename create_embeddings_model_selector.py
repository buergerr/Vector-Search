import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import pinecone
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from model_utils import initialize_model_and_tokenizer, move_model_to_device, model_options
import re

# Download NLTK stopwords
nltk.download('stopwords')

# Load environment variables from .env file
load_dotenv()

# Get the Pinecone API key from environment variables
PINECONE_APIKEY = os.getenv('PINECONE_APIKEY')

# Define the file path for the CSV file
file_path = 'file.csv'

# Try reading the CSV file with different encodings
encodings = ['utf-8', 'latin1', 'iso-8859-1']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc, on_bad_lines='skip')
        print(f"Successfully read the CSV file with encoding: {enc}")
        break
    except Exception as e:
        print(f"Failed to read the CSV file with encoding: {enc}. Error: {e}")

if df is not None:
    print(df.head())
    print(df.columns)

    # Clean up the column names
    df.columns = [col.strip() for col in df.columns]

    # Load German stopwords
    stop_words = set(stopwords.words('german'))

    # Function to remove stopwords
    def remove_stopwords(text):
        tokens = text.split()
        tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(tokens)

    import re

    # Function to normalize text
    def normalize_text(text):
        # Convert text to lowercase
        text = text.lower()
        
        # Replace German special characters with ASCII equivalents
        text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        text = text.replace('ß', 'ss')
        
        # Remove special characters but preserve letters, numbers, and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


    # Preprocess the text to remove stopwords and normalize text. Combine multiple columns into one.
    df['combined_text'] = df[['shortdescrdisplay','longdescrdisplay', 'searchcolor']].apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1)
    df['combined_text'] = df['combined_text'].apply(remove_stopwords)
    df['combined_text'] = df['combined_text'].apply(normalize_text)

    # Print some of the processed text to see how it looks before embedding
    print("Sample processed text before embedding:")
    print(df['combined_text'].head(5))

    # Choose the model you want to use (e.g., the first one)
    selected_model = model_options[5]

    # Initialize the selected model and tokenizer
    tokenizer, model = initialize_model_and_tokenizer(selected_model)

    # Move model to GPU if available, else use CPU
    device = move_model_to_device(model)

    # Function to create embeddings in batches
    def create_embeddings(text_list, batch_size=16):
        embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size), desc="Creating embeddings"):
            batch_texts = text_list[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.extend(batch_embeddings)
        return embeddings

    # Ensure combined_text is not empty or invalid
    df = df.dropna(subset=['combined_text'])

    # Create embeddings for the combined text
    embeddings = create_embeddings(df['combined_text'].tolist())

    # Print some sample embeddings to inspect
    print("\nSample embeddings:")
    for i, embedding in enumerate(embeddings[:1]):  # Print the first 5 embeddings
        print(f"Embedding {i+1}: {embedding}")

    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=PINECONE_APIKEY)

    # After creating the embeddings, check their dimension
    embedding_dim = embeddings[0].shape[0]

    # Adjust Pinecone index creation based on the embedding dimension
    index_name = 'xlm-roberta-base-de-en' # should have less then 45 characters and no special characters. Onyl "-" is allowed
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric='cosine',  # Change this to 'cosine' for cosine similarity
            spec=pinecone.ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Connect to the index
    index = pc.Index(index_name)

    # Convert metadata fields to appropriate types for Pinecone
    df['sku'] = df['sku'].astype(str)
    df['shortdescrdisplay'] = df['shortdescrdisplay'].astype(str)
    df['manufacturer'] = df['manufacturer'].astype(str)
    df['searchcolor'] = df['searchcolor'].astype(str)
    df['imageurl'] = df['imageurl'].astype(str)

    # Create list of metadata and vectors
    vectors = [
        (
            str(i),
            embedding,
            {
                'sku': df.iloc[i]['sku'],
                'shortdescrdisplay': df.iloc[i]['shortdescrdisplay'],                
                'manufacturer': df.iloc[i]['manufacturer'],
                'searchcolor': df.iloc[i]['searchcolor'],
                'imageurl': df.iloc[i]['imageurl'],            
            }
        )
        for i, embedding in enumerate(embeddings)
    ]

    # Upsert the vectors into the Pinecone index in smaller batches with progress display
    batch_size = 100
    for i in tqdm(range(0, len(vectors), batch_size), desc="Storing embeddings in Pinecone"):
        batch_vectors = vectors[i:i+batch_size]
        ids, embeds, metas = zip(*batch_vectors)
        index.upsert(vectors=zip(ids, embeds, metas))

    print("Embeddings and metadata stored in Pinecone successfully!")
else:
    print("Failed to load the CSV file with any encoding.")
