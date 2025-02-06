import warnings
warnings.filterwarnings('ignore')

import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
import time

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
import PyPDF2
import tempfile
from shutil import copyfile

# Constants
DOCS_DIR = 'docs'
PERSIST_DIRECTORY = 'vector_store'
HISTORY_FILE = 'chat_history.json'
SUPPORTED_EXTENSIONS = ['.pdf', '.txt']

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# for azure openai
import base64
import requests
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv(override=True)

# Get environment variables without defaults
app_key = os.getenv('app_key')  # app key is required
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

# Validate required environment variables
if not all([langsmith_api_key, app_key, client_id, client_secret]):
    raise ValueError("Missing required environment variables. Please check your .env file contains: app_key, client_id, client_secret, LANGSMITH_API_KEY")

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization to avoid warnings

def init_openai():
    """Initialize Azure OpenAI with hardcoded credentials"""
    url = "https://id.cisco.com/oauth2/default/v1/token"
    payload = "grant_type=client_credentials"
    value = base64.b64encode(f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')
    headers = {
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {value}",
        "User": f'{{"appkey": "{app_key}"}}'
    }
    
    token_response = requests.request("POST", url, headers=headers, data=payload)
    
    llm = AzureChatOpenAI(
        azure_endpoint='https://chat-ai.cisco.com',
        api_key=token_response.json()["access_token"],
        api_version="2023-08-01-preview",
        temperature=0,
        max_tokens=1000,
        model="gpt-4o",
        model_kwargs={
            "user": f'{{"appkey": "{app_key}"}}'
        }
    )
    return llm, token_response.json()

def convert_pdf_to_text(pdf_path):
    """
    Convert a PDF file to text using PyPDF2.
    
    Args:
        pdf_path (Path): Path to the PDF file
    Returns:
        str: Extracted text content
    """
    try:
        logger.info(f"Starting PDF conversion for: {pdf_path}")
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            logger.info(f"PDF loaded with {len(reader.pages)} pages")
            text = "\n".join([page.extract_text() for page in reader.pages])
            logger.info(f"Text extracted, length: {len(text)} characters")
        return text
    except Exception as e:
        logger.error(f"Error converting PDF to text: {str(e)}")
        raise

def process_uploaded_file(uploaded_file):
    """
    Process an uploaded file, converting PDFs to text if necessary.
    
    Args:
        uploaded_file: StreamlitUploadedFile object
    Returns:
        str: Path to the processed file in docs directory
    """
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        logger.info(f"Processing uploaded file: {uploaded_file.name} ({file_extension})")
        
        if file_extension == '.pdf':
            logger.info("PDF file detected, starting conversion process")
            # Create temporary file to store PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                tmp_pdf.write(uploaded_file.getbuffer())
                tmp_pdf_path = tmp_pdf.name
                logger.info(f"Temporary PDF saved at: {tmp_pdf_path}")
            
            # Convert PDF to text
            text_content = convert_pdf_to_text(tmp_pdf_path)
            
            # Create text file in docs directory
            text_filename = Path(uploaded_file.name).stem + '.txt'
            text_filepath = os.path.join(DOCS_DIR, text_filename)
            logger.info(f"Saving text to: {text_filepath}")
            
            # Save text content
            with open(text_filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Clean up temporary PDF file
            os.unlink(tmp_pdf_path)
            logger.info("Temporary PDF file cleaned up")
            
            return text_filepath
            
        elif file_extension == '.txt':
            # Save text file directly to docs directory
            file_path = os.path.join(DOCS_DIR, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            return file_path
            
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
            
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        raise

def process_documents(progress_bar=None):
    """Process documents and create/update vector store"""
    try:
        # Create directories first
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
        
        # Load and split documents
        logger.info("Loading documents...")
        
        # Initialize document loaders for different file types
        loaders = {
            '.pdf': PyPDFDirectoryLoader(DOCS_DIR),
            '.txt': DirectoryLoader(
                DOCS_DIR,
                glob="**/*.txt",
                loader_cls=lambda file_path: TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
            )
        }
        
        # Configure text splitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=512,
            chunk_overlap=16
        )
        
        # Process all document types
        all_chunks = []
        for ext, loader in loaders.items():
            try:
                chunks = loader.load_and_split(text_splitter)
                logger.info(f"Loaded {len(chunks)} chunks from {ext} files")
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error loading {ext} files: {str(e)}")
        
        if progress_bar:
            progress_bar.progress(0.5, "Documents loaded and split")
        
        if len(all_chunks) > 0:
            # Create/Update vector store with all documents
            logger.info("Creating/Updating vector store...")
            vector_store = Chroma.from_documents(
                documents=all_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            logger.info("Vector store updated successfully")
            
            # Test persistence by creating a new instance
            logger.info("Testing persistence...")
            persisted_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_model
            )
            logger.info("Persistence verified successfully")
            
            if progress_bar:
                progress_bar.progress(1.0, "Vector store created and verified")
            
            return vector_store
        else:
            logger.warning("No documents found to process")
            return None
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise

def RAG(query, vector_store, retriever):
    try:
        # Get relevant chunks and log them in detail
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Query: '{query}'")
        
        # Remove duplicate chunks by content
        unique_docs = []
        seen_content = set()
        for doc in docs:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_docs.append(doc)
        
        logger.info(f"Retrieved {len(unique_docs)} unique chunks")
        logger.info("Retrieved chunks content:")
        for i, doc in enumerate(unique_docs):
            logger.info(f"\nChunk {i+1}:")
            logger.info(f"Source: {doc.metadata.get('source', 'unknown')}")
            logger.info(f"Content: {doc.page_content}")

        # Format context to be more explicit
        context_list = []
        for doc in unique_docs:
            content = doc.page_content.strip()
            source = Path(doc.metadata.get('source', '')).name
            content = f"From {source}:\n{content}"
            context_list.append(content)
        
        context_for_query = "\n\n---\n\n".join(context_list)

        # More explicit system message
        qna_system_message = """
You are an assistant whose primary task is to answer questions about Cisco products and technologies.
When answering questions:
1. Look for ALL relevant information across ALL provided documents
2. Consider the conversation history for context
3. If this is a follow-up question, relate it to previous exchanges
4. Combine information from different sources to provide a complete answer
5. Include technical details, specifications, and features when available
6. Mention if information comes from different documents
7. Use information even if it's marked as internal or draft
8. If you find multiple pieces of information about the topic, organize them under appropriate headers
9. Always end your response with a "Document Reference" section that lists which documents contained the information

Format your response in clear markdown with:
- Main header for the topic
- Subheaders for different aspects (Overview, Features, Specifications, etc.)
- Bold text for important terms
- Bullet points for features or details
- Proper spacing between sections
- Final section titled "Document Reference" explaining which documents were used
"""

        # Format conversation history if it exists
        conversation_history = ""
        if hasattr(st.session_state, 'conversation_context') and st.session_state.conversation_context:
            history_items = []
            for item in st.session_state.conversation_context[-3:]:  # Last 3 exchanges
                history_items.append(f"User: {item['question']}")
                history_items.append(f"Assistant: {item['answer']}")
            conversation_history = "\n".join(history_items)
            logger.info(f"Using conversation history with {len(history_items)//2} exchanges")

        # User message template that includes both context and history
        qna_user_message_template = """
Here is the context containing information about Cisco products:

{context}

Previous conversation:
{history}

Based on the above context and conversation history, please answer this question:
{question}
"""

        # Format the messages
        messages = [
            {"role": "system", "content": qna_system_message},
            {"role": "user", "content": qna_user_message_template.format(
                context=context_for_query,
                history=conversation_history,
                question=query
            )}
        ]

        # Initialize OpenAI (using existing BIT initialization)
        llm = init_openai()[0]  # Note: BIT version returns tuple of (llm, token_response)

        # Query the LLM with generation parameters
        try:
            response = llm.invoke(
                messages,
                temperature=0,
                top_p=0.9,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=2000
            )
            
            # Update conversation context
            if not hasattr(st.session_state, 'conversation_context'):
                st.session_state.conversation_context = []
            
            st.session_state.conversation_context.append({
                'question': query,
                'answer': response.content
            })
            
            # Keep only the last 5 exchanges
            if len(st.session_state.conversation_context) > 5:
                st.session_state.conversation_context = st.session_state.conversation_context[-5:]
            
            return response.content

        except Exception as e:
            logger.error(f"Error in RAG function: {str(e)}")
            return f'Sorry, I encountered the following error: \n {e}'

    except Exception as e:
        logger.error(f"Error in RAG function: {str(e)}")
        return f'Sorry, I encountered the following error: \n {e}'

def save_chat_history(history):
    """Save chat history to file"""
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}")

def load_chat_history():
    """Load chat history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading chat history: {str(e)}")
    return []

def handle_history():
    st.header("Conversation History")
    
    # Header with Clear All button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.subheader("Past Conversations")
    with col2:
        if st.button("🗑️ Clear All", 
                  type="secondary", 
                  use_container_width=True,
                  help="Delete conversation history",
                  key="clear_history"):
            # Only clear chat history file and history state
            st.session_state.chat_history = []
            save_chat_history([])
            st.rerun()  # Just rerun to refresh UI
    
    if st.session_state.chat_history:
        for idx, item in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Show question and delete button in the same row
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.info(f"**Q:** {item['question']}")
                with col2:
                    if st.button("🗑️", 
                             key=f"del_hist_{idx}", 
                             help="Delete this conversation"):
                        # Remove item from history
                        st.session_state.chat_history.remove(item)
                        save_chat_history(st.session_state.chat_history)
                        st.rerun()
                
                # Show answer in collapsible section
                with st.expander("Show Answer", expanded=False):
                    st.markdown(f"**🕒 {item['timestamp']}**")
                    st.markdown(item['answer'])
                st.divider()
    else:
        st.info("No conversation history available.")

def main():
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG Assistant")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    if 'current_chat' not in st.session_state:
        st.session_state.current_chat = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'last_processed' not in st.session_state:
        st.session_state.last_processed = None
    if 'documents_initialized' not in st.session_state:
        st.session_state.documents_initialized = False

    # Only initialize documents if not already done
    if not st.session_state.documents_initialized:
        # Check for both PDF and TXT files
        docs = []
        for ext in SUPPORTED_EXTENSIONS:
            docs.extend(list(Path(DOCS_DIR).glob(f'*{ext}')))
        if docs:
            try:
                # Check if we have a persisted vector store first
                embedding_model = HuggingFaceEmbeddings(model_name='thenlper/gte-large')
                if os.path.exists(PERSIST_DIRECTORY):
                    logger.info("Loading existing vector store...")
                    st.session_state.vector_store = Chroma(
                        persist_directory=PERSIST_DIRECTORY,
                        embedding_function=embedding_model
                    )
                    st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    logger.info("Creating new vector store...")
                    st.session_state.vector_store = process_documents()
                    if st.session_state.vector_store:
                        st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error(f"Error processing documents: {str(e)}")
                st.session_state.vector_store = None
        st.session_state.documents_initialized = True
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "📝 History"])
    
    # Documents Tab
    with tab2:
        st.header("Document Management")
        
        # File uploader with support for both PDF and text files
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Select PDF or text files to upload",
            key="doc_uploader"
        )
        
        if uploaded_files:
            with st.status("Processing documents...", expanded=True) as status:
                try:
                    # Process each uploaded file
                    for file in uploaded_files:
                        try:
                            status.write(f"Starting to process: {file.name}")
                            processed_path = process_uploaded_file(file)
                            os.chmod(processed_path, 0o666)  # Make files readable/writable
                            status.write(f"✅ Processed: {Path(processed_path).name}")
                            # Add a check to verify the file exists
                            if os.path.exists(processed_path):
                                status.write(f"✓ Verified file exists: {processed_path}")
                            else:
                                status.write(f"❌ Warning: File not found: {processed_path}")
                        except Exception as e:
                            status.write(f"❌ Error processing {file.name}: {str(e)}")
                            logger.error(f"Error details for {file.name}: {str(e)}", exc_info=True)
                            continue
                    
                    # Process all documents in one go
                    progress_bar = st.progress(0)
                    st.session_state.vector_store = process_documents(progress_bar)
                    if st.session_state.vector_store:
                        status.update(label="✅ Documents processed successfully!", state="complete")
                        st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        status.update(label="⚠️ No documents were processed", state="error")
                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)}", state="error")
                    st.error(f"Error processing documents: {str(e)}")
                    st.session_state.vector_store = None
        
        # Show current documents
        st.divider()
        
        # Get all documents
        docs = []
        for ext in SUPPORTED_EXTENSIONS:
            docs.extend(list(Path(DOCS_DIR).glob(f'*{ext}')))
        
        # Show header with document count
        st.subheader(f"Current Documents ({len(docs)})")
        
        if docs:
            # Show last processed time if available
            if 'last_processed' in st.session_state and st.session_state.last_processed:
                st.info(f"🕒 Last processed: {st.session_state.last_processed}")
            
            # Document list
            for doc in docs:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.text(doc.name)
                with col2:
                    st.text(f"({doc.suffix})")
                with col3:
                    if st.button("🗑️", key=f"del_{doc.name}"):
                        try:
                            # Get absolute path to ensure proper file handling
                            file_path = os.path.abspath(doc)
                            
                            # Try to delete the file
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                time.sleep(0.5)  # Give OS time to complete deletion
                                
                                if not os.path.exists(file_path):
                                    logger.info(f"Successfully deleted file: {doc.name}")
                                    
                                    # Clear file uploader state
                                    if "doc_uploader" in st.session_state:
                                        del st.session_state["doc_uploader"]
                                    
                                    # Clear caches
                                    st.cache_data.clear()
                                    st.cache_resource.clear()
                                    
                                    # Process remaining documents and update vector store
                                    try:
                                        st.session_state.vector_store = process_documents()
                                        if st.session_state.vector_store:
                                            st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                            st.success(f"Deleted {doc.name} and updated vector store")
                                        else:
                                            st.warning("No documents remaining after deletion")
                                    except Exception as e:
                                        st.error(f"Error updating vector store after deletion: {str(e)}")
                                        logger.error(f"Error updating vector store: {str(e)}")
                                        st.session_state.vector_store = None
                                        st.session_state.last_processed = None
                                    
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete {doc.name}. File still exists.")
                            else:
                                st.error(f"File not found: {doc.name}")
                        except Exception as e:
                            logger.error(f"Error deleting file {doc.name}: {str(e)}")
                            st.error(f"Error deleting file {doc.name}: {str(e)}")
            
            # Add Process Documents button
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🔄 Process Documents", use_container_width=True):
                    with st.status("Processing documents...", expanded=True) as status:
                        try:
                            # Reset vector store state
                            st.session_state.vector_store = None
                            st.session_state.last_processed = None
                            st.session_state.documents_initialized = False
                            
                            # Clear embedding model cache
                            st.cache_resource.clear()
                            
                            progress_bar = st.progress(0)
                            status.write("Creating new vector store...")
                            
                            # Process documents with fresh state
                            st.session_state.vector_store = process_documents(progress_bar)
                            if st.session_state.vector_store:
                                st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                status.update(label="✅ Documents processed successfully!", state="complete")
                                st.session_state.documents_initialized = True
                            else:
                                status.update(label="⚠️ No documents were processed", state="error")
                        except Exception as e:
                            status.update(label=f"❌ Error: {str(e)}", state="error")
                            st.error(f"Error processing documents: {str(e)}")
                            # Clean up failed state
                            st.session_state.vector_store = None
                            st.session_state.last_processed = None
                            st.session_state.documents_initialized = False
        else:
            st.info("No documents uploaded yet")
    
    # Chat Tab
    with tab1:
        st.header("Chat with Documents")
        
        # Add a button to start new conversation at the top
        col1, col2 = st.columns([4, 1])
        with col2:
            if st.button("🔄 Start New Chat", help="Clear current conversation and start fresh", key="new_chat", use_container_width=True):
                # Clear all conversation-related state
                st.session_state.current_chat = []
                st.session_state.conversation_context = []
                # Don't clear chat_history as that's for the History tab
                
                # Force a complete context reset
                if 'messages' in st.session_state:
                    del st.session_state.messages
                
                # Clear any other conversation-related cache
                st.cache_data.clear()
                
                st.rerun()  # Rerun to refresh UI
        
        # Check if documents exist
        docs = []
        for ext in SUPPORTED_EXTENSIONS:
            docs.extend(list(Path(DOCS_DIR).glob(f'*{ext}')))
        if not docs:
            st.warning("Please upload documents in the Documents tab first")
        elif st.session_state.vector_store is None:
            st.warning("Please process your documents in the Documents tab first")
        else:
            # Create a container for the chat interface
            chat_container = st.container()
            
            # Display current conversation
            with chat_container:
                if st.session_state.current_chat:
                    for chat in st.session_state.current_chat:
                        with st.chat_message("user"):
                            st.write(chat['question'])
                        with st.chat_message("assistant"):
                            st.markdown(chat['answer'])
            
            # Empty space to push the chat input to the bottom
            st.markdown("<br>" * 2, unsafe_allow_html=True)
            
            # Place chat input at the bottom
            if not st.session_state.current_chat:
                # Initial question
                query = st.chat_input("Ask a new question about your documents...", key="chat_input")
            else:
                # Follow-up question
                query = st.chat_input("Ask a follow-up question...", key="chat_input")
            
            if query:
                # Add user message to chat
                with chat_container:
                    with st.chat_message("user"):
                        st.write(query)
                
                with st.status("Thinking...", expanded=True) as status:
                    try:
                        # Use existing vector store - no reprocessing needed
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type='mmr',  # Changed from 'similarity' to 'mmr' for more diverse results
                            search_kwargs={
                                'k': 15,  # Increased from 12
                                'fetch_k': 20  # Fetch more candidates for MMR to choose from
                            }
                        )
                        
                        response = RAG(
                            query, 
                            st.session_state.vector_store,
                            retriever=retriever
                        )
                        
                        # Save to history and current chat
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        chat_item = {
                            "timestamp": timestamp,
                            "question": query,
                            "answer": response
                        }
                        st.session_state.chat_history.append(chat_item)
                        st.session_state.current_chat.append(chat_item)
                        save_chat_history(st.session_state.chat_history)
                        
                        # Display assistant response
                        with chat_container:
                            with st.chat_message("assistant"):
                                st.markdown(response)
                                
                        # Force a rerun to update the chat interface
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
                        logger.error(f"Error in RAG: {str(e)}")
    
    # History Tab
    with tab3:
        handle_history()

if __name__ == "__main__":
    main() 