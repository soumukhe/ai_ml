import warnings
warnings.filterwarnings('ignore')

import json
import tiktoken
import logging
import pandas as pd
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAI
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set OpenAI logger to WARNING to suppress HTTP request logs
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv(override=True)

# Get environment variables without defaults
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Validate required environment variables
if not all([langsmith_api_key, OPENAI_API_KEY]):
    raise ValueError("Missing required environment variables. Please check your .env file contains: app_key, client_id, client_secret, LANGSMITH_API_KEY, and OPENAI_API_KEY")

# Initialize OpenAI
def init_openai():
    """Initialize OpenAI with API key"""
    llm = ChatOpenAI(
        model="gpt-4o",
        verbose=False
    )
    return llm

def load_pdf_documents(docs_dir: str) -> List:
    """
    Load PDF documents from the specified directory with error handling
    """
    try:
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            raise FileNotFoundError(f"Directory not found: {docs_dir}")
        
        logger.info(f"Loading PDF files from: {docs_dir}")
        files = list(docs_path.glob('*.pdf'))
        logger.info(f"Found PDF files: {[f.name for f in files]}")
        
        pdf_loader = PyPDFDirectoryLoader(docs_dir)
        
        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap=16
        )
        
        doc_chunks = pdf_loader.load_and_split(text_splitter)
        logger.info(f"Successfully loaded {len(doc_chunks)} document chunks")
        return doc_chunks
        
    except Exception as e:
        logger.error(f"Error loading PDF documents: {str(e)}")
        raise

def get_document_hash(docs_dir: str) -> str:
    """
    Create a hash of document names and their modification times to detect changes
    """
    doc_info = []
    docs_path = Path(docs_dir)
    for file in sorted(docs_path.glob('*.pdf')):  # sort for consistency
        mtime = os.path.getmtime(file)
        doc_info.append(f"{file.name}:{mtime}")
    return "|".join(doc_info)

def save_document_hash(hash_str: str, persist_directory: str):
    """Save document hash to a file"""
    hash_file = Path(persist_directory) / "doc_hash.txt"
    hash_file.write_text(hash_str)

def load_document_hash(persist_directory: str) -> str:
    """Load document hash from file"""
    hash_file = Path(persist_directory) / "doc_hash.txt"
    return hash_file.read_text() if hash_file.exists() else ""

def get_document_info(docs_dir: str) -> dict:
    """
    Get document information including name, modification time, and size
    Returns a dictionary with filename as key and [mtime, size] as value
    """
    doc_info = {}
    docs_path = Path(docs_dir)
    for file in docs_path.glob('*.pdf'):
        # Round mtime to avoid floating point comparison issues
        mtime = round(os.path.getmtime(file), 2)
        size = os.path.getsize(file)
        doc_info[file.name] = [mtime, size]  # Use list instead of tuple for JSON compatibility
    return doc_info

def save_document_info(doc_info: dict, persist_directory: str):
    """Save document info to a file"""
    info_file = Path(persist_directory) / "doc_info.json"
    with open(info_file, 'w') as f:
        json.dump(doc_info, f)

def load_document_info(persist_directory: str) -> dict:
    """Load document info from file"""
    info_file = Path(persist_directory) / "doc_info.json"
    if info_file.exists():
        with open(info_file) as f:
            return json.load(f)
    return {}

def get_changed_documents(current_info: dict, stored_info: dict) -> tuple:
    """
    Compare current and stored document info to find changes
    Returns (new_files, modified_files, deleted_files)
    """
    new_files = set(current_info.keys()) - set(stored_info.keys())
    deleted_files = set(stored_info.keys()) - set(current_info.keys())
    
    # Compare files that exist in both
    modified_files = set()
    for fname in current_info.keys() & stored_info.keys():
        current_mtime, current_size = current_info[fname]
        stored_mtime, stored_size = stored_info[fname]
        
        # Compare rounded values
        if round(current_size) != round(stored_size) or \
           round(current_mtime, 2) != round(stored_mtime, 2):
            modified_files.add(fname)
    
    return new_files, modified_files, deleted_files

def load_specific_pdfs(docs_dir: str, filenames: set) -> List:
    """Load specific PDF files and split them into chunks"""
    if not filenames:
        return []
    
    docs_path = Path(docs_dir)
    specific_files = [str(docs_path / fname) for fname in filenames]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=512,
        chunk_overlap=16
    )
    
    all_chunks = []
    for file in specific_files:
        loader = PyPDFLoader(file)
        chunks = loader.load_and_split(text_splitter)
        all_chunks.extend(chunks)
    
    return all_chunks

docs_dir = 'docs'
doc_chunks = load_pdf_documents(docs_dir)
print(f"Total number of document chunks: {len(doc_chunks)}")

# embedding model
embedding_model = HuggingFaceEmbeddings(model_name='thenlper/gte-large')

# Check if vector store exists and handle document changes
persist_directory = 'vector_store'
current_info = get_document_info(docs_dir)
should_recreate = True

if os.path.exists(persist_directory):
    stored_info = load_document_info(persist_directory)
    new_files, modified_files, deleted_files = get_changed_documents(current_info, stored_info)
    
    if not any([new_files, modified_files, deleted_files]):
        logger.info("Loading existing vector store (no document changes detected)...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        logger.info(f"Loaded vector store with {vector_store._collection.count()} documents")
        should_recreate = False
    else:
        if new_files:
            logger.info(f"New files detected: {new_files}")
        if modified_files:
            logger.info(f"Modified files detected: {modified_files}")
        if deleted_files:
            logger.info(f"Deleted files detected: {deleted_files}")
            
        # Load existing vector store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
        
        # Handle deletions
        if deleted_files:
            # Note: This is a placeholder as Chroma doesn't have a direct way to delete by source.
            # For now, we'll recreate the entire store if files are deleted
            logger.info("Deleted files detected, need to recreate entire vector store")
            should_recreate = True
        else:
            should_recreate = False
            # Process new and modified files
            changed_files = new_files | modified_files
            if changed_files:
                logger.info(f"Processing changed files: {changed_files}")
                new_chunks = load_specific_pdfs(docs_dir, changed_files)
                if new_chunks:
                    logger.info(f"Adding {len(new_chunks)} new chunks to vector store")
                    vector_store.add_documents(new_chunks)

if should_recreate:
    if os.path.exists(persist_directory):
        logger.info("Removing old vector store...")
        import shutil
        shutil.rmtree(persist_directory)
    
    logger.info("Creating new vector store...")
    doc_chunks = load_pdf_documents(docs_dir)
    vector_store = Chroma.from_documents(
        documents=doc_chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    logger.info(f"Created new vector store with {len(doc_chunks)} documents")

# Save current document info
save_document_info(current_info, persist_directory)

# create retriever
retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)

# test vector store retreiver

# user_input = "What is ROCE ?"

# relevant_document_chunks = retriever.get_relevant_documents(user_input)

# print(len(relevant_document_chunks))
# print("--------------------------------")

# for document in relevant_document_chunks:
#     relevant_document = document.page_content.replace("\t", " ")

# print(relevant_document)

# defining the llm
llm = init_openai()

# Definging templates
## defining the qna system message
qna_system_message = """
You are an assistant whose work is to review the report and provide the appropriate answers from the context.
User input will have the context required by you to answer user questions.
This context will begin with the token: ###Context.
The context contains references to specific portions of a document relevant to the user query.

User questions will begin with the token: ###Question.

Please answer only using the context provided in the input. Do not mention anything about the context in your final answer.

please include benefits and use cases when appropriate to answer the question.

If the answer is not found in the context, respond "I don't know".
"""

## defining the qna user message template
qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Question
{question}
"""

# Defining the rag system:

def RAG(user_input):
    """
    Args:
    user_input: Takes a user input for which the response should be retrieved from the vectorDB.
    Returns:
    relevant context as per user query.
    """
    # Use invoke instead of get_relevant_documents
    relevant_document_chunks = retriever.invoke(user_input)
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)

    # Format the messages properly for ChatOpenAI
    messages = [
        {"role": "system", "content": qna_system_message},
        {"role": "user", "content": qna_user_message_template.format(context=context_for_query, question=user_input)}
    ]

    # Query the LLM with generation parameters
    try:
        response = llm.invoke(
            messages,
            temperature=0,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=1000
        )
        return response.content

    except Exception as e:
        return f'Sorry, I encountered the following error: \n {e}'

print("response:")
print(RAG("what is ROCE ?  Please give detailed answers in a nicely formatted markdown"))


