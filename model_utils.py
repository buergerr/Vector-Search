from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertModel
import torch

# List of available models
model_options = [
    'distilbert-base-uncased',
    'sentence-transformers/all-MiniLM-L12-v2',
    'Sakil/sentence_similarity_semantic_search',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'Sahajtomar/German-semantic',
    'PM-AI/sts_paraphrase_xlm-roberta-base_de-en' # (5) is working
]

def initialize_model_and_tokenizer(model_name):
    if model_name == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def move_model_to_device(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return device
