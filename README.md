
# Real-Time Search Application with Embeddings and Pinecone Integration

This project demonstrates a real-time search application that utilizes pre-trained transformer models for generating embeddings from textual data and integrates with Pinecone for efficient vector search. The application is built using Python, leveraging frameworks like Flask for the web interface and various NLP libraries for text processing and embedding creation.

## Features

- **Text Preprocessing**: Cleans and normalizes text data, removes stopwords, and handles special characters.
- **Embedding Creation**: Uses transformer models to generate embeddings for text data.
- **Pinecone Integration**: Efficiently stores and retrieves embeddings for real-time search.
- **Real-Time Search**: Allows users to search for similar items based on query embeddings.


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
    Create a `.env` file in the root directory and configure the following environment variables:
    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_INDEX_HOST=your_pinecone_index_host  # Optional if not using a specific host
    PINECONE_INDEX_NAME=your_index_name  # Must be all lowercase, up to 45 characters, with no special characters except '-'
    PRODUCT_IMG_BASE_URL=your_image_base_url  # Base URL for product images
    PRODUCT_IMG_SUFFIX=your_image_suffix  # Optional suffix for image URLs
    CSV_FILE_PATH=your_csv_file_path  # Path to your CSV file containing the data
    ```

    Example of a `.env` file:
    ```env
    PINECONE_API_KEY=abc123xyz
    PINECONE_INDEX_HOST=us-east-1-aws.pinecone.io
    PINECONE_INDEX_NAME=product-search-index
    PRODUCT_IMG_BASE_URL=https://images.example.com/products/
    PRODUCT_IMG_SUFFIX=?imwidth=384
    CSV_FILE_PATH=data/products.csv
    ```
    ```

4. **Download NLTK stopwords**:
    Run the following command to download the necessary NLTK data:
    ```python
    python -c "import nltk; nltk.download('stopwords')"
    ```

## Embedding Instructions

1. **Configure Your Environment**:
   - Ensure your `.env` file is properly set up with the correct Pinecone API key, index details, and paths as described above.

2. **Run the Embedding Script**:
   - Execute the script to preprocess your data, generate embeddings, and store them in Pinecone:
     ```bash
     python preprocess_and_embed.py
     ```
   - This script will:
     - Read the specified CSV file.
     - Preprocess the text data, including cleaning and normalizing.
     - Generate embeddings using a selected pre-trained transformer model.
     - Store the embeddings in the specified Pinecone index.

3. **Verify Embedding Storage**:
   - Once the embeddings are stored, you can verify them by accessing your Pinecone dashboard or using Pinecone's CLI tools.

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
    - The search results will be ranked based on cosine similarity.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

