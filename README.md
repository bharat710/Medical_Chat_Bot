# Medical Chatbot using Llama2

A Retrieval-Augmented Generation (RAG) based medical chatbot application that leverages the Llama2 Large Language Model to provide accurate and context-aware answers to medical queries. The system uses LangChain for orchestration and Pinecone as a vector database for efficient semantic search.

## Features

- **Interactive Chat Interface**: A user-friendly web interface built with Flask for real-time interaction.
- **Unified Data Ingestion**: Supports uploading PDF documents and ingesting content from URLs directly via the API.
- **RAG Architecture**: Combines the power of Llama2 (via CTransformers) with a knowledge base stored in Pinecone.
- **Efficient Retrieval**: Uses HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) for semantic similarity search.

## Tech Stack

- **Language**: Python 3.8+
- **LLM**: Llama-2-7b-chat (GGML format)
- **Framework**: LangChain, Flask
- **Vector Database**: Pinecone
- **Embeddings**: HuggingFace (`sentence-transformers`)
- **Frontend**: HTML/CSS/JS (Templates)

## Prerequisites

Before running the application, ensure you have the following:

1.  **Python 3.8 or higher** installed.
2.  **Pinecone API Key**: Sign up at [Pinecone](https://www.pinecone.io/) and create an index (e.g., named `testing`).
3.  **Llama2 Model**: Download the `llama-2-7b-chat.ggmlv3.q4_0.bin` model file.
    *   Place it in the `model/` directory.

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/Medical-chatbot-using-Llama2.git
    cd Medical-chatbot-using-Llama2
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    conda create -n mchatbot python=3.8 -y
    conda activate mchatbot
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Environment Variables**
    Create a `.env` file in the root directory and add your Pinecone credentials:

    ```env
    PINECONE_API_KEY=your_pinecone_api_key
    PINECONE_API_ENV=your_pinecone_environment_region
    ```

2.  **Model Setup**
    Ensure the Llama2 model binary is located at `model/llama-2-7b-chat.ggmlv3.q4_0.bin`. You can configure the model path in `app.py` if different.

## Usage

### 1. Ingesting Data
You can ingest data (PDFs or URLs) to populate your Pinecone vector index.

*   **Option A: Using the Script**
    Edit `store_index.py` to point to your data directory or specific URLs, then run:
    ```bash
    python store_index.py
    ```

*   **Option B: Via API/UI**
    The application supports an `/ingest` endpoint to upload files or URLs dynamically.

### 2. Running the Chatbot
Start the Flask application:

```bash
python app.py
```

The application will run on `http://localhost:8080`. Open this URL in your browser to start chatting with the bot.

## Project Structure

```
├── app.py               # Main Flask application and unique routes
├── store_index.py       # Script for data ingestion and vectorization
├── src/
│   ├── helper.py        # Helper functions for loading data and embeddings
│   └── prompt.py        # Prompt templates for the LLM
├── templates/
│   └── chat.html        # Chat interface HTML template
├── static/              # Static assets (CSS/JS)
├── data/                # Directory for storing uploaded/source PDFs
├── model/               # Directory for the Llama2 binary model
├── .env                 # Environment variables (API keys)
├── requirements.txt     # Python dependencies
└── setup.py             # Project setup configuration
```