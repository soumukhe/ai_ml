#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import json
import shutil
import os
import chromadb
import time


def read_config(config_file):
  """Reads a JSON config file and returns a dictionary."""
  with open(config_file, 'r') as f:
    return json.load(f)

# Read configuration
config = read_config("config1_api.json")
api_key = config.get("AZURE_OPENAI_KEY")
endpoint = config.get("AZURE_OPENAI_ENDPOINT")
llamaparse_api_key = config.get("LLAMA_KEY")
groq_api_key = config.get("GROQ_API_KEY")

# Initialize embeddings
embedding_model = OpenAIEmbeddings(
    openai_api_base=endpoint,
    openai_api_key=api_key,
    model='text-embedding-ada-002'
)

# Define paths
persisted_vectordb_location = './rag_db'
collection_name = 'semantic_chunks'

def ensure_directory_permissions(directory):
    """Ensure the directory has proper write permissions."""
    if os.path.exists(directory):
        # Change permissions to ensure write access
        os.chmod(directory, 0o777)
        for root, dirs, files in os.walk(directory):
            for d in dirs:
                os.chmod(os.path.join(root, d), 0o777)
            for f in files:
                os.chmod(os.path.join(root, f), 0o777)

def clean_vector_store():
    # First, check if the directory exists
    if not os.path.exists(persisted_vectordb_location):
        print(f"No vector store found at {persisted_vectordb_location}")
        return

    try:
        # Ensure proper permissions
        ensure_directory_permissions(persisted_vectordb_location)
        
        # Initialize Chroma client to access the database directly
        client = chromadb.PersistentClient(path=persisted_vectordb_location)
        
        try:
            # Get the collection
            collection = client.get_collection(collection_name)
            
            # Get the count of documents
            count = collection.count()
            print(f"Found {count} documents in the vector store")
            
            if count > 0:
                # Get all document IDs
                results = collection.get()
                if results and 'ids' in results:
                    # Delete documents by their IDs
                    collection.delete(ids=results['ids'])
                    print(f"Deleted {len(results['ids'])} documents from the vector store")
                else:
                    print("No document IDs found to delete")
            else:
                print("Vector store is empty")
        except Exception as e:
            print(f"Error accessing collection: {str(e)}")
        finally:
            # Close the client
            client = None
            
        # Small delay to ensure all file handles are released
        time.sleep(1)
        
        # Remove the directory to ensure complete cleanup
        shutil.rmtree(persisted_vectordb_location)
        print(f"Removed vector store directory at {persisted_vectordb_location}")
        
    except Exception as e:
        print(f"Error cleaning vector store: {str(e)}")
        # If there's an error, try to remove the directory anyway
        if os.path.exists(persisted_vectordb_location):
            shutil.rmtree(persisted_vectordb_location)
            print(f"Force removed vector store directory at {persisted_vectordb_location}")

# Clean the vector store first
clean_vector_store()

# Create directory with proper permissions
os.makedirs(persisted_vectordb_location, exist_ok=True)
ensure_directory_permissions(persisted_vectordb_location)

# Create a new Chroma instance
print("\nCreating new Chroma vector store...")
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=persisted_vectordb_location,
    embedding_function=embedding_model
)

print("Vector store initialized successfully!")

# Safely clear all documents from the collection
def clear_vector_store(vector_store):
    # Check if there are any documents to delete
    collection_data = vector_store.get()

    if "ids" in collection_data and collection_data["ids"]:
        # Delete all documents if there are any
        print(f"Deleting {len(collection_data['ids'])} documents from the vector store...")
        vector_store.delete(ids=collection_data["ids"])
        print("Vector store cleared successfully.")
    else:
        print("Vector store is already empty. Nothing to delete.")

# Usage
clear_vector_store(vector_store)