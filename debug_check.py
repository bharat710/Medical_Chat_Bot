from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
import pinecone
from pinecone import Pinecone

# 1. Check Env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
env = os.getenv("PINECONE_API_ENV")

print(f"API Key present: {bool(api_key)}")
print(f"Env present: {bool(env)}")

if not api_key:
    print("CRITICAL: PINECONE_API_KEY missing in .env")
    exit(1)

# 2. Check Model File
model_path = "model/llama-2-7b-chat.ggmlv3.q4_0.bin"
if os.path.exists(model_path):
    print(f"Model file found at {model_path}")
else:
    print(f"CRITICAL: Model file NOT found at {model_path}")

# 3. Check Pinecone Index
try:
    pc = Pinecone(api_key=api_key)
    formatted_indexes = [i.name for i in pc.list_indexes()]
    print(f"Available indexes: {formatted_indexes}")
    
    index_name = "testing"
    if index_name not in formatted_indexes:
        print(f"CRITICAL: Index '{index_name}' not found!")
        # Check if 'scholar-pulse' exists
        if "scholar-pulse" in formatted_indexes:
            print("Suggest: Update code to use 'scholar-pulse'.")
    else:
        print(f"Index '{index_name}' found.")
        idx = pc.Index(index_name)
        stats = idx.describe_index_stats()
        print(f"Index Stats: {stats}")
        
        if stats['total_vector_count'] == 0:
            print("CRITICAL: Index is EMPTY.")
        else:
            # 4. Try Retrieval
            print("Attempting retrieval...")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
            docs = docsearch.similarity_search("What are the symptoms of fever?", k=2)
            print(f"Retrieval returned {len(docs)} documents.")
            for i, doc in enumerate(docs):
                print(f"Doc {i}: {doc.page_content[:100]}...")

except Exception as e:
    print(f"Error connecting to Pinecone: {e}")
