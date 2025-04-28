import json
import platform
import torch
import time  # Add time module import
import requests
import os

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def check_gpu_availability():
    """Check if GPU is available and return appropriate device."""
    if platform.system() == 'Darwin':  # macOS
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print("Metal GPU acceleration is available")
            return "mps"  # Return string instead of torch.device
        else:
            print("Metal GPU acceleration is not available")
    elif torch.cuda.is_available():
        print("CUDA GPU acceleration is available")
        return "cuda"
    print("Using CPU")
    return "cpu"

# Set the device for PyTorch operations
DEVICE = check_gpu_availability()
print(f"Selected device: {DEVICE}")

def read_config(config_file):
  """Reads a JSON config file and returns a dictionary."""
  with open(config_file, 'r') as f:
    return json.load(f)

config = read_config("config1_api.json")  #Copy and paste the path of the config file uploaded in Colab
api_key = config.get("AZURE_OPENAI_KEY")
endpoint = config.get("AZURE_OPENAI_ENDPOINT") # Remove the trailing comma here
llamaparse_api_key = config.get("LLAMA_KEY")
groq_api_key = config.get("GROQ_API_KEY")


import os
import chromadb

from dotenv import load_dotenv
import json

from langchain_core.documents import Document
#from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_openai import ChatOpenAI
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from llama_index.core import Settings, SimpleDirectoryReader



from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)


# # Initialize the OpenAI Embeddings with device awareness
# embedding_model = OpenAIEmbeddings(
#     openai_api_base=endpoint,
#     openai_api_key=api_key,
#     model='text-embedding-ada-002',
#     model_kwargs={"device": device}  # Pass device through model_kwargs instead
# )
# # This initializes the OpenAI embeddings model using the specified endpoint, API key, and model name.

# Initialize local embedding model
from sentence_transformers import SentenceTransformer
import torch

# Initialize the model with GPU support
model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2",
    device=DEVICE
)
print(f"Model loaded on device: {model.device}")

# Create a wrapper class for SentenceTransformer to work with SemanticChunker
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model
        self.device = DEVICE
    
    def embed_documents(self, texts):
        # Convert single string to list if necessary
        if isinstance(texts, str):
            texts = [texts]
        
        # Get embeddings from the model
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=True
            )
            # Move to CPU for numpy conversion
            if self.device != "cpu":
                embeddings = embeddings.cpu()
            return embeddings.numpy().tolist()
    
    def embed_query(self, text):
        return self.embed_documents(text)[0]

# Create embeddings instance
embeddings = SentenceTransformerEmbeddings(model)

# Create a custom embedding function for Chroma
def embed_function(texts):
    # Convert single string to list if necessary
    if isinstance(texts, str):
        texts = [texts]
    
    # Process in batches for better performance
    batch_size = 32
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Keep everything on GPU until the final conversion
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                device=DEVICE,
                show_progress_bar=True
            )
            # Move to CPU only if necessary
            if DEVICE != "cpu":
                embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings.numpy())
    
    return all_embeddings

# Create a wrapper class for the embedding function
class EmbeddingWrapper:
    def __init__(self, embed_func):
        self.embed_func = embed_func
        self.device = DEVICE
    
    def embed_documents(self, texts):
        return self.embed_func(texts)
    
    def embed_query(self, text):
        if isinstance(text, str):
            text = [text]
        return self.embed_func(text)[0]  # Return first embedding for single query

# Initialize the embedding wrapper
embedding_wrapper = EmbeddingWrapper(embed_function)

# # Initialize the Chat OpenAI model
# llm = ChatOpenAI(
#     openai_api_base=endpoint,
#     openai_api_key=api_key,
#     model="gpt-4o-mini",
#     streaming=False
# )

# defining bridgeit llm
from dotenv import load_dotenv
import json

# Force reload of environment variables
load_dotenv(override=True)

# Get environment variables without defaults
app_key = os.getenv('app_key')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')


from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import logging
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Validate required environment variables
if not all([app_key, client_id, client_secret, langsmith_api_key]):
    raise ValueError("Missing required environment variables")


# Configuration parameters
CONFIG = {
    # Agent settings
    'agent': {
        'max_iterations': 5,
        'max_execution_time': 110.0,
        'temperature': 0,
        'max_tokens': 4000,
    },
    
    # Retry settings
    'retry': {
        'max_retries': 2,
        'sleep_seconds': 4,
    },
    
    # Duplicate detection settings
    'duplicates': {
        'similarity_threshold': 0.95,  # Restored to original value for stricter duplicate detection
    },
    
    # Sentiment analysis settings
    'sentiment': {
        'labels': ["highRating+", "highRating", "Neutral", "Negative", "Negative-"],
        'hypothesis_template': "This feature request is {}."
    },
    
    # Batch processing settings
    'batch': {
        'gpu_batch_size': 128,
        'cpu_batch_size': 32,
    }
}



## BridgeIT Azure OpenAI

def init_azure_openai():
    """Initialize Azure OpenAI with hardcoded credentials and token refresh"""
    def get_fresh_token():
        url = "https://id.cisco.com/oauth2/default/v1/token"
        payload = "grant_type=client_credentials"
        value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {value}",
            "User": f'{{"appkey": "{app_key}"}}'
        }
        
        try:
            token_response = requests.request("POST", url, headers=headers, data=payload)
            token_response.raise_for_status()  # Raise exception for bad status codes
            return token_response.json()["access_token"] #return the access token
        except Exception as e:
            logger.error(f"Error getting token: {str(e)}")
            raise

    # Try to get a fresh token with retries
    max_retries = CONFIG['retry']['max_retries']
    for attempt in range(max_retries):
        try:
            access_token = get_fresh_token()
            
            llm = AzureChatOpenAI(
                azure_endpoint='https://chat-ai.cisco.com',
                api_key=access_token,
                api_version="2023-08-01-preview",
                temperature=0,
                max_tokens=16000,  # Increased from 4000 to 8000
                model="gpt-4o-mini",
                model_kwargs={
                    "user": f'{{"appkey": "{app_key}"}}'
                }
            )
            
            # Test the connection
            test_response = llm.invoke("Test connection")
            if test_response:
                return llm
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(CONFIG['retry']['sleep_seconds'])
                continue
            raise Exception(f"Failed to initialize Azure OpenAI after {max_retries} attempts: {str(e)}")
            
    raise Exception("Failed to initialize Azure OpenAI")


#llm = init_azure_openai()




# Initialize semantic chunker with our wrapped embeddings
semantic_text_splitter = SemanticChunker(
    embeddings,  # Use our wrapped embeddings
    # breakpoint_threshold_type='gradient',
    breakpoint_threshold_type='percentile',
    breakpoint_threshold_amount=85
)

from langchain_community.document_loaders import UnstructuredPowerPointLoader

# List to store parsed JSON objects
json_objs = []

# Define the folder containing the documents
folder_path = "ppt"

# Iterate through PPTs in the folder and parse content
for ppt in os.listdir(folder_path):
    if ppt.endswith(".pptx"):
        ppt_path = os.path.join(folder_path, ppt)
        print(f"\nProcessing file: {ppt}")
        
        try:
            # Use UnstructuredPowerPointLoader
            loader = UnstructuredPowerPointLoader(ppt_path)
            for doc in loader.load():
                # Ensure source is set to the PPT filename
                doc.metadata['source'] = ppt
                json_objs.append({"page_content": doc.page_content, "metadata": doc.metadata})
            print(f"Successfully processed: {ppt}")
                
        except Exception as e:
            print(f"Error processing {ppt}: {str(e)}")
            continue

print(f"\nTotal documents processed: {len(json_objs)}")

# Convert JSON objects to Document objects for semantic chunking
data = [Document(page_content=obj["page_content"], metadata=obj["metadata"]) for obj in json_objs]

# Process documents with semantic chunker
semantic_chunks = []
for doc in data:
    chunks = semantic_text_splitter.split_text(doc.page_content)
    # Ensure we preserve the document name in metadata
    doc_metadata = doc.metadata.copy()
    if 'source' in doc_metadata:
        doc_metadata['document_name'] = doc_metadata['source']
    semantic_chunks.extend([Document(page_content=chunk, metadata=doc_metadata) for chunk in chunks])

print(f"Total number of chunks: {len(semantic_chunks)}")


# Add IDs to the semantic chunks
semantic_chunks = [Document(id=i, page_content=d.page_content, metadata=d.metadata) for i, d in enumerate(semantic_chunks)]

# # Plot a histogram of the number of characters in each semantic chunk
# import matplotlib.pyplot as plt

# chunk_lengths = [len(chunk.page_content) for chunk in semantic_chunks]
# plt.hist(chunk_lengths, bins=25, edgecolor='black')
# plt.xlabel('Chunk Length')
# plt.ylabel('Frequency')
# plt.title('Number of Characters in Each Semantic Chunk')
# plt.show()


# Collection and persistence settings
persisted_vectordb_location = './rag_db'
collection_name = 'semantic_chunks'

# Initialize the Chroma vector store with the wrapped embedding function
vector_store = Chroma(
    collection_name=collection_name,
    persist_directory=persisted_vectordb_location,
    embedding_function=embedding_wrapper  # Use the wrapper instead of raw function
)

# Process documents in smaller batches
batch_size = 50  # Reduced batch size for better memory management
for i in range(0, len(semantic_chunks), batch_size):
    batch = semantic_chunks[i : i + batch_size]
    try:
        vector_store.add_documents(batch)
        print(f"Processed batch {i//batch_size + 1}/{(len(semantic_chunks) + batch_size - 1)//batch_size}")
    except Exception as e:
        print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
        continue

print("Vector store updated successfully!")

# Verify metadata preservation
print("\nVerifying metadata preservation in vector store...")
sample_docs = vector_store.similarity_search("", k=1)  # Get one sample document
if sample_docs:
    print("\nSample document metadata:")
    print(json.dumps(sample_docs[0].metadata, indent=2))
else:
    print("No documents found in vector store")

# Define prompt for generating hypothetical questions
hypothetical_questions_prompt = """Based on the following document content:
{doc}

Generate exactly 10 hypothetical questions that a curious reader might ask to explore the topic further.
The questions should be open-ended, insightful, and designed to prompt deeper discussion and analysis of key themes,
details, and implications within the content.

Format your response as a numbered list of exactly 10 questions."""

import time

# List to store documents with hypothetical questions
hyp_questions = []
sleep_interval = 20  # Pause after every 20 chunks to avoid rate limits
sleep_duration = 10  # Pause duration in seconds (2 minutes) - this is the rate limit cooldown period

# Generate hypothetical questions for each semantic chunk
llm = init_azure_openai()
for i, document in enumerate(semantic_chunks):
    # Insert a pause after every 'sleep_interval' iterations
    if i > 0 and i % sleep_interval == 0:
        print(f"Processed {i}/{len(semantic_chunks)} chunks; sleeping for {sleep_duration} seconds to avoid rate limit...")
        time.sleep(sleep_duration)

    try:
        # Add a small delay between each API call
        if i > 0:  # Don't sleep before the first call
            time.sleep(5)  # 5 second delay between calls
            
        # Invoke the LLM to generate questions based on the chunk content
        response = llm.invoke(hypothetical_questions_prompt.format(doc=document.page_content))
        # Access content as a string instead of a bound method
        questions = response.content  # Extract the generated questions
        print(f"Generated questions for chunk {i+1}/{len(semantic_chunks)}")
    except Exception as e:
        print(f"Error generating questions for chunk {i+1}: {str(e)}")
        if "Rate limit" in str(e):
            print("Rate limit hit, waiting 30 seconds before retry...")
            time.sleep(30)
            continue
        questions = "No questions generated"  # Assign a default value if generation fails

    # Create metadata for the generated questions
    metadata = {
        'original_content': document.page_content,
        'source': document.metadata.get('source', 'N/A'),
        'document_name': document.metadata.get('document_name', 'N/A'),  # Add document name
        'page': document.metadata.get('page', -1),
        'type': 'hypothetical_questions',
        'original_chunk_id': document.id  # Add the original chunk ID
    }

    # Create and store the document containing generated questions
    hyp_questions.append(
        Document(
            id=str(i),            # Assign a unique ID to each generated document
            page_content=questions,  # Store the generated questions
            metadata=metadata       # Attach metadata
        )
    )

    if i == len(semantic_chunks) - 1:
        print(f"Processed {i + 1}/{len(semantic_chunks)} chunks.")

## do 1 time only
import pickle

# To save hyp_questions to a file:
with open("hyp_questions.pkl", "wb") as f:
    pickle.dump(hyp_questions, f)

## to reload hyp_questions:
import pickle

with open("hyp_questions.pkl", "rb") as f:
    hyp_questions = pickle.load(f)

# Function to print a sample document with hypothetical questions
def print_sample(docs, index=0):
    # Check if the index is within the bounds of the list
    if 0 <= index < len(docs):
        print("ID:\n", docs[index].id, "\n")
        print("Metadata:")
        print(json.dumps(docs[index].metadata, indent=4), "\n")
        print("Hypothetical Questions:\n", docs[index].page_content)
    else:
        print(f"Index {index} is out of range for the list with length {len(docs)}.")

# Print a sample document, ensuring the index is valid
print_sample(hyp_questions, index=min(20, len(hyp_questions) - 1))  # Adjust index if necessary    

# Store hypothetical questions in Chroma vector store
print("\nStoring hypothetical questions in vector store...")
vector_store_hyp = Chroma(
    collection_name="hypothetical_questions",
    persist_directory=persisted_vectordb_location,
    embedding_function=embedding_wrapper
)

# Process hypothetical questions in batches
batch_size = 50
for i in range(0, len(hyp_questions), batch_size):
    batch = hyp_questions[i : i + batch_size]
    try:
        vector_store_hyp.add_documents(batch)
        print(f"Processed hypothetical questions batch {i//batch_size + 1}/{(len(hyp_questions) + batch_size - 1)//batch_size}")
    except Exception as e:
        print(f"Error processing hypothetical questions batch {i//batch_size + 1}: {str(e)}")
        continue

print("Hypothetical questions stored successfully!")

# Print samples from vector database to verify storage
print("\nVerifying vector database samples...")
print("\nSample from semantic chunks collection:")
sample_docs = vector_store.similarity_search("", k=3)  # Get 3 sample documents
for i, doc in enumerate(sample_docs, 1):
    print(f"\nDocument {i}:")
    print("ID:", doc.id)
    print("Metadata:")
    print(json.dumps(doc.metadata, indent=4))
    print("Content preview:", doc.page_content[:200] + "...")  # Show first 200 characters

# If we have hypothetical questions stored in a separate collection, show those too
try:
    hyp_collection_name = "hypothetical_questions"
    vector_store_hyp = Chroma(
        collection_name=hyp_collection_name,
        persist_directory=persisted_vectordb_location,
        embedding_function=embedding_wrapper
    )
    print("\nSample from hypothetical questions collection:")
    hyp_sample_docs = vector_store_hyp.similarity_search("", k=3)
    for i, doc in enumerate(hyp_sample_docs, 1):
        print(f"\nHypothetical Questions Document {i}:")
        print("ID:", doc.id)
        print("Metadata:")
        print(json.dumps(doc.metadata, indent=4))
        print("Questions:", doc.page_content)
except Exception as e:
    print("\nNo hypothetical questions collection found or error accessing it:", str(e))    