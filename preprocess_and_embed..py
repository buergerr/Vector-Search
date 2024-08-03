import os
import pandas as pd
from dotenv import load_dotenv
import torch
from model_utils import initialize_model_and_tokenizer, move_model_to_device, model_options
from embedding_utils import remove_stopwords, normalize_text, create_embeddings, initialize_pinecone, upsert_vectors

# Load environment variables
load_dotenv()

PINECONE_APIKEY = os.getenv('PINECONE_API_KEY')
file_path = os.environ.get('CSV_FILE_PATH')
index_name = os.environ.get('PINECONE_INDEX_NAME')

selected_model = model_options[3] # Choose the model you want to use

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

# Preprocess the text
df['combined_text'] = df[['shortdescrdisplay','longdescrdisplay', 'searchcolor']].apply(
    lambda row: ' '.join(row.values.astype(str)), axis=1)
df['combined_text'] = df['combined_text'].apply(lambda x: remove_stopwords(x, language='german'))
df['combined_text'] = df['combined_text'].apply(normalize_text)

# Initialize the model and tokenizer
tokenizer, model = initialize_model_and_tokenizer(selected_model)
device = move_model_to_device(model)

# Check if the model is on the GPU
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
else:
    print("Using CPU")

# Create embeddings
embeddings = create_embeddings(df['combined_text'].tolist(), tokenizer, model, device)

# Initialize Pinecone client
pc = initialize_pinecone(PINECONE_APIKEY)

# Try to upsert vectors
try:
    upsert_vectors(index_name, pc, df, embeddings, embeddings[0].shape[0])
    print("Embeddings and metadata stored in Pinecone successfully!")
except Exception as e:
    print(f"Failed to store embeddings in Pinecone: {e}")
    new_index_name = input("Please enter a new index name to retry (max 45 character, no speacial characters are allowed, only '-'): ")
    try:
        upsert_vectors(new_index_name, pc, df, embeddings, embeddings[0].shape[0])
        print(f"Embeddings and metadata stored in Pinecone successfully in the new index: {new_index_name}")
    except Exception as retry_exception:
        print(f"Retry failed: {retry_exception}. Please check the Pinecone index name and configuration.")
