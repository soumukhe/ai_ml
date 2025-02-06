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
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
DOCS_DIR = 'docs'
PERSIST_DIRECTORY = 'vector_store'
HISTORY_FILE = 'chat_history.json'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv(override=True)

# Get environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in .env file")

# Set tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallel tokenization to avoid warnings

def init_openai():
    """Initialize OpenAI with API key"""
    return ChatOpenAI(
        model="gpt-4o",
        verbose=False
    )

def process_documents(progress_bar=None):
    """Process documents and create/update vector store"""
    try:
        # Create directories first
        os.makedirs(DOCS_DIR, exist_ok=True)
        os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
        
        if progress_bar:
            progress_bar.progress(0.1, "Initializing embedding model (instructor-xl)...")
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name='hkunlp/instructor-xl',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if progress_bar:
            progress_bar.progress(0.2, "Loading and splitting documents...")
        
        # Load and split documents
        logger.info("Loading documents...")
        pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name='cl100k_base',
            chunk_size=1000,
            chunk_overlap=200
        )
        
        doc_chunks = pdf_loader.load_and_split(text_splitter)
        chunk_count = len(doc_chunks)
        logger.info(f"Loaded {chunk_count} document chunks")
        
        if progress_bar:
            progress_bar.progress(0.3, f"Processing {chunk_count} document chunks...")
        
        if len(doc_chunks) > 0:
            # Create/Update vector store with all documents
            logger.info("Creating/Updating vector store...")
            if progress_bar:
                progress_bar.progress(0.4, "Creating embeddings (this may take several minutes)...")
                status = st.empty()
                status.info("⏳ Creating embeddings with instructor-xl model. This is a one-time process and may take several minutes. The larger embedding model provides better accuracy but requires more processing time.")
            
            vector_store = Chroma.from_documents(
                documents=doc_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            
            if progress_bar:
                progress_bar.progress(0.8, "Verifying vector store...")
                status.success("✅ Embeddings created successfully!")
            
            logger.info("Vector store updated successfully")
            
            # Test persistence by creating a new instance
            logger.info("Testing persistence...")
            if progress_bar:
                progress_bar.progress(0.9, "Testing persistence...")
            
            persisted_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_model
            )
            logger.info("Persistence verified successfully")
            
            if progress_bar:
                progress_bar.progress(1.0, "Processing complete!")
            
            return vector_store
        else:
            logger.warning("No documents found to process")
            if progress_bar:
                progress_bar.error("No documents found to process")
            return None
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        if progress_bar:
            progress_bar.error(f"Error: {str(e)}")
        raise

def RAG(user_input, vector_store, retriever=None, llm=None):
    """
    Args:
    user_input: Takes a user input for which the response should be retrieved from the vectorDB.
    Returns:
    relevant context as per user query.
    """
    # Append formatting request to user input
    formatted_input = f"{user_input} Please give detailed answers in a nicely formatted markdown"
    
    # Always use the provided retriever
    if retriever is None:
        raise ValueError("Retriever must be provided")
    
    if llm is None:
        llm = init_openai()

    # Get relevant documents using the provided retriever
    relevant_document_chunks = retriever.get_relevant_documents(formatted_input)
    
    # Improve context combination with better formatting and separation
    context_list = []
    for i, doc in enumerate(relevant_document_chunks, 1):
        # Clean the content and add section numbering
        content = doc.page_content.replace('\n', ' ').strip()
        # Add metadata if available
        source_info = f" [Source: {doc.metadata.get('source', 'Unknown')}]" if doc.metadata.get('source') else ""
        context_list.append(f"[{i}] {content}{source_info}")
    
    # Join contexts with clear separation
    context_for_query = "\n\n".join(context_list)

    # System message
    qna_system_message = """
You are an expert assistant tasked with providing accurate and detailed answers based on the provided context.
The context contains numbered sections [1], [2], etc., each representing a relevant portion of the documents.

When answering:
1. Use information from ALL relevant context sections
2. Combine information from different sections when they complement each other
3. If the answer is partially found, provide what you can find and indicate what aspects are not covered
4. Only respond with "I don't know" if absolutely no relevant information is found in any context section

Format your response in clear, well-structured markdown with:
- Appropriate headers using # for main sections
- Bullet points for lists and key points
- Bold text for important terms using **term**
- Code blocks or quotes where relevant
- Tables if presenting structured data
- Numbered lists for sequential information
- Proper spacing between sections

Include benefits and use cases when they are mentioned in the context.
"""

    # User message template with conversation history
    qna_user_message_template = """
###Context
Here are some documents that are relevant to the question mentioned below.
{context}

###Conversation History
{history}

###Question
{question}
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

    # Format the messages properly for ChatOpenAI
    messages = [
        {"role": "system", "content": qna_system_message},
        {"role": "user", "content": qna_user_message_template.format(
            context=context_for_query,
            history=conversation_history,
            question=user_input
        )}
    ]

    # Query the LLM with generation parameters
    try:
        response = llm.invoke(
            messages,
            temperature=0,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=4000
        )
        
        # Update conversation context
        if not hasattr(st.session_state, 'conversation_context'):
            st.session_state.conversation_context = []
            
        st.session_state.conversation_context.append({
            'question': user_input,
            'answer': response.content
        })
        
        # Keep only the last 5 exchanges
        if len(st.session_state.conversation_context) > 5:
            st.session_state.conversation_context = st.session_state.conversation_context[-5:]
        
        return response.content

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
        docs = list(Path(DOCS_DIR).glob('*.pdf'))
        if docs:
            try:
                # Show initial loading status
                status = st.empty()
                progress_bar = st.progress(0)
                status.info("🚀 Initializing RAG Assistant...")
                
                # Check if we have a persisted vector store first
                embedding_model = HuggingFaceEmbeddings(
                    model_name='hkunlp/instructor-xl',
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                if os.path.exists(PERSIST_DIRECTORY):
                    status.info("📚 Loading existing vector store...")
                    progress_bar.progress(0.3)
                    st.session_state.vector_store = Chroma(
                        persist_directory=PERSIST_DIRECTORY,
                        embedding_function=embedding_model
                    )
                    progress_bar.progress(1.0)
                    status.success("✅ Vector store loaded successfully!")
                    st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    status.info("🔄 Creating new vector store (this may take several minutes)...")
                    st.session_state.vector_store = process_documents(progress_bar)
                    if st.session_state.vector_store:
                        st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        status.success("✅ Initial setup complete!")
                    
                st.session_state.documents_initialized = True
                time.sleep(1)  # Give UI time to update
                st.rerun()  # Refresh to show all tabs
                
            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")
                logger.error(f"Error processing documents: {str(e)}")
                st.session_state.vector_store = None
        st.session_state.documents_initialized = True
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📚 Documents", "📝 History"])
    
    # Documents Tab
    with tab2:
        st.header("Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload",
            key="pdf_uploader"  # Add a specific key
        )
        
        if uploaded_files:
            with st.status("Processing documents...", expanded=True) as status:
                try:
                    # Save all uploaded files with proper permissions
                    for file in uploaded_files:
                        file_path = os.path.join(DOCS_DIR, file.name)
                        with open(file_path, 'wb') as f:
                            f.write(file.getbuffer())
                        os.chmod(file_path, 0o666)  # Make files readable/writable
                        status.write(f"Saved: {file.name}")
                    
                    # Process all documents in one go
                    progress_bar = st.progress(0)
                    st.session_state.vector_store = process_documents(progress_bar)
                    if st.session_state.vector_store:
                        status.update(label="✅ Documents processed successfully!", state="complete")
                        st.session_state.last_processed = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state.documents_initialized = True  # Set flag after successful processing
                    else:
                        status.update(label="⚠️ No documents were processed", state="error")
                except Exception as e:
                    status.update(label=f"❌ Error: {str(e)}", state="error")
                    st.error(f"Error processing documents: {str(e)}")
                    st.session_state.vector_store = None
        
        # Show current documents
        st.divider()
        st.subheader("Current Documents")
        docs = list(Path(DOCS_DIR).glob('*.pdf'))
        if docs:
            # Show last processed time if available
            if 'last_processed' in st.session_state and st.session_state.last_processed:
                st.info(f"🕒 Last processed: {st.session_state.last_processed}")
            
            # Document list
            for doc in docs:
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(doc.name)
                with col2:
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
                                    if "pdf_uploader" in st.session_state:
                                        del st.session_state["pdf_uploader"]
                                    
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
                st.session_state.current_chat = []
                st.session_state.conversation_context = []
                st.rerun()  # Just rerun, don't clear vector store
        
        # Check if documents exist
        docs = list(Path(DOCS_DIR).glob('*.pdf'))
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
                            search_type='similarity',
                            search_kwargs={
                                'k': 8  # Number of relevant chunks to retrieve
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