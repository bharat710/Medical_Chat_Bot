from flask import Flask, render_template, jsonify, request
from langchain import PromptTemplate
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

embeddings = download_hugging_face_embeddings()

index_name = "testing"

docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

PROMPT = PromptTemplate(template = prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

# from langchain.vectorstores.base import VectorStoreRetriever
llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8,
                          'context_length':4096})

qa = RetrievalQA.from_chain_type(  
    llm=llm,  
    chain_type="stuff",  
    retriever=docsearch.as_retriever(    
        search_type="similarity",
        search_kwargs={"k": 5}),
    chain_type_kwargs=chain_type_kwargs 
) 

from store_index import load_data, get_text_chunks
from langchain_community.embeddings import HuggingFaceEmbeddings # Re-importing here just to be safe if not picked from helper or store_index context properly, though usually shared. Actually, better to use the same logic.

# Ensure upload directory exists
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Re-use embeddings from global scope logic or ensure consistent model
# embeddings = download_hugging_face_embeddings() # Already imported and initialized

@app.route('/ingest', methods=['POST'])
def ingest():
    try:
        files = request.files.getlist('file')
        url = request.form.get('url')
        
        saved_files = []
        if files:
            for file in files:
                if file.filename:
                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(file_path)
                    saved_files.append(file_path)
        
        urls = [url] if url else []
        
        if not saved_files and not urls:
            return "No data provided", 400

        print(f"Ingesting: Files={saved_files}, URLs={urls}")
        
        # Load Data
        # We only want to load the NEWLY added data to append to index, 
        # BUT load_data as written in store_index.py loads EVERYTHING in 'data/' directory.
        # This is inefficient if 'data/' grows large, but fits the "Unified Ingestion" request.
        # However, for 'Just In Time', user might expect just this doc.
        # If I use `load_data` with `data/`, it re-indexes everything.
        # Modified approach: modify load_data call or standard logic to only load specific new files?
        # `PyPDFDirectoryLoader` loads directory.
        # For this implementation to be robust without refactoring store_index heavily:
        # I'll rely on the fact user asked to "upload document ... which act as data".
        # If I re-index everything, it duplicates vectors unless I handle IDs.
        # Pinecone upsert overwrites if same ID, LangChain default IDs are content-based often or UUIDs? 
        # PineconeVectorStore.from_documents usually generates new IDs.
        # Duplicate content = duplicate search results.
        # Ideally, we should ingest ONLY the new file.
        
        # Refined Logic:
        # 1. If URL, load URL.
        # 2. If File, load Just that file? PyPDFLoader (singular) instead of Directory.
        # But I need to import PyPDFLoader. `store_index.py` has `PyPDFDirectoryLoader`.
        # I will import PyPDFLoader here.
        
        docs = []
        
        # Handle Files
        from langchain_community.document_loaders import PyPDFLoader
        for file_path in saved_files:
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
            
        # Handle URLs
        if urls:
            from langchain_community.document_loaders import WebBaseLoader
            loader = WebBaseLoader(urls)
            docs.extend(loader.load())
            
        if not docs:
             return "No content extracted", 400
             
        # Split
        text_chunks = get_text_chunks(docs)
        print(f"Split into {len(text_chunks)} chunks")
        
        # Embed and Push
        # docsearch is already initialized as a helper for retrieval, 
        # but to add documents we can use .add_documents() on the existing vectorstore object if available,
        # OR PineconeVectorStore.from_documents with same index name.
        
        PineconeVectorStore.from_documents(
            text_chunks, 
            embeddings, 
            index_name=index_name
        )
        
        return f"Successfully ingested {len(docs)} documents ({len(text_chunks)} chunks).", 200
        
    except Exception as e:
        print(e)
        return str(e), 500

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg.strip()
    print(input)
    result = qa({"query": input})
    print("Response:", result['result'])
    return str(result['result'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080, debug=True)

