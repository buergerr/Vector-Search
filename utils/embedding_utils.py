import re
import torch
from tqdm import tqdm
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast
import pinecone
from pinecone import Pinecone, ServerlessSpec

# Text preprocessing functions
def remove_stopwords(text, language='german'):
    stop_words = set(stopwords.words(language))
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def normalize_text(text):
    text = text.lower()
    text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    text = text.replace('ß', 'ss')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Embedding creation function
def create_embeddings(text_list, tokenizer, model, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc="Creating embeddings"):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            with autocast(device_type='cuda'):
                outputs = model(**inputs)
        
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return embeddings


# Pinecone operations
def initialize_pinecone(api_key):
    pc = Pinecone(api_key=api_key)
    return pc

def upsert_vectors(index_name, pc, df, embeddings, embedding_dim, batch_size=100):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    index = pc.Index(index_name)

    df['sku'] = df['sku'].astype(str)
    df['shortdescrdisplay'] = df['shortdescrdisplay'].astype(str)
    df['manufacturer'] = df['manufacturer'].astype(str)
    df['searchcolor'] = df['searchcolor'].astype(str)
    df['imageurl'] = df['imageurl'].astype(str)

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

    for i in tqdm(range(0, len(vectors), batch_size), desc="Storing embeddings in Pinecone"):
        batch_vectors = vectors[i:i+batch_size]
        ids, embeds, metas = zip(*batch_vectors)
        index.upsert(vectors=zip(ids, embeds, metas))
