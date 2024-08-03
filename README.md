
# Real-Time Search Application with Embeddings and Pinecone Integration

This project demonstrates a real-time search application that utilizes pre-trained transformer models for generating embeddings from textual data and integrates with Pinecone for efficient vector search. The application is built using Python, leveraging frameworks like Flask for the web interface and various NLP libraries for text processing and embedding creation.

## Features

- **Text Preprocessing**: Cleans and normalizes text data, removes stopwords, and handles special characters.
- **Embedding Creation**: Uses transformer models to generate embeddings for text data.
- **Pinecone Integration**: Efficiently stores and retrieves embeddings for real-time search.
- **Real-Time Search**: Allows users to search for similar items based on query embeddings.
- **NDCG Calculation**: Evaluates search relevance using Normalized Discounted Cumulative Gain (NDCG).

## Requirements

- Python 3.8+
- Libraries and tools required:
  - `os`
  - `pandas`
  - `torch`
  - `nltk`
  - `tqdm`
  - `re`
  - `flask`
  - `numpy`
  - `sklearn`
  - `transformers`
  - `pinecone`
  - `python-dotenv`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Install dependencies**:
    You can install the required Python packages using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up environment variables**:
    Create a `.env` file in the root directory and add your Pinecone API key:
    ```env
    PINECONE_APIKEY=your_pinecone_api_key
    ```

4. **Download NLTK stopwords**:
    Run the following command to download the necessary NLTK data:
    ```python
    python -c "import nltk; nltk.download('stopwords')"
    ```

## Usage

1. **Preprocess and Embed Text Data**:
    - Run the script that preprocesses the text data and creates embeddings:
      ```bash
      python preprocess_and_embed.py
      ```
    - This script will read a CSV file, preprocess the text data, generate embeddings using a pre-trained transformer model, and store the embeddings in a Pinecone index.

2. **Start the Real-Time Search Web App**:
    - Launch the Flask application:
      ```bash
      python search_web_app_realtime.py
      ```
    - Access the web app in your browser at `http://localhost:5000`.

3. **Search for Similar Items**:
    - Use the web interface to input a search query and find items with similar embeddings in the Pinecone index.
    - The search results will be ranked based on cosine similarity, and the NDCG metric will be calculated to evaluate the relevance of the results.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

