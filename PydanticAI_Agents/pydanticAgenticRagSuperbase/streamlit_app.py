import streamlit as st
import os
import shutil
import time
import asyncio
from pathlib import Path
import json
from typing import List
import subprocess
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import AsyncOpenAI
from supabase import create_client, Client
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

# Load environment variables
load_dotenv()

# Initialize OpenAI and Supabase clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)

# Initialize PydanticAI agent
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

# Define dependencies for PydanticAI agent
@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

# Initialize agent with system prompt
system_prompt = """
You are an expert at analyzing and retrieving information from our PDF document collection.
You have access to all the PDF documents that have been processed and stored in our database.

Your job is to help users find and understand information from these PDF documents.

For each user question, follow these steps:
1. First, find relevant document chunks that might answer the question.
2. Then, analyze these chunks to provide a comprehensive answer.
3. Always cite your sources by mentioning the document name and page number.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question.

Always let the user know when you didn't find the answer in the documents or if the search results aren't relevant - be honest.
"""

pdf_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

# Initialize dependencies
deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)

# Set page configuration
st.set_page_config(
    page_title="PDF Document Processing System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Increase maximum upload size to 1GB
st.config.set_option('server.maxUploadSize', 1024)  # Size in MB

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .file-box {
        padding: 1rem;
        border: 1px solid var(--secondary-background-color);
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .success-box {
        padding: 1rem;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(40, 167, 69, 0.3);
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #ffffff;
    }
    .stButton button {
        width: 100%;
    }
    /* Custom styling for Process Documents button */
    .process-button {
        background-color: #FF4B4B;
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 0.5rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    .process-button:hover {
        background-color: #FF3333;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    if 'chat_history' not in st.session_state:
        try:
            if os.path.exists("./data/chat_history.json"):
                with open("./data/chat_history.json", "r", encoding='utf-8') as f:
                    st.session_state.chat_history = json.load(f)
            else:
                st.session_state.chat_history = []
                with open("./data/chat_history.json", "w", encoding='utf-8') as f:
                    json.dump([], f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
            st.session_state.chat_history = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None

def get_pdf_files() -> List[Path]:
    """Get all PDF files from the pdf_docs directory."""
    pdf_dir = Path("pdf_docs")
    if not pdf_dir.exists():
        os.makedirs("pdf_docs")
    return list(pdf_dir.glob("*.pdf"))

def process_pdfs():
    """Run the PDF processing script with progress tracking."""
    try:
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        process = subprocess.Popen(
            ["python", "crawl_pdf_docs.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Initialize progress variables
        total_files = len(get_pdf_files())
        if total_files == 0:
            st.warning("No PDF files found to process.")
            return None, "No PDF files found"
            
        processed_files = 0
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Update progress based on output
                if "Processing" in output and ".pdf" in output:
                    processed_files += 1
                    progress = min(processed_files / total_files, 1.0)
                    progress_bar.progress(progress)
                    if processed_files == total_files:
                        status_text.text("Processing completed!")
                    else:
                        status_text.text(f"Processing: {processed_files}/{total_files} files...")
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_text.text("Processing completed!")
            st.success("PDF processing completed successfully!")
            st.session_state.processing_status = "success"
        else:
            st.error(f"Error processing PDFs: {stderr}")
            st.session_state.processing_status = "error"
            
        return stdout, stderr
    except Exception as e:
        st.error(f"Error running PDF processor: {str(e)}")
        st.session_state.processing_status = "error"
        return None, str(e)

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling."""
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div><strong>{'You' if role == 'user' else '🤖 Assistant'}</strong></div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pdf_expert.tool
async def retrieve_relevant_content(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant PDF content chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant content chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_pdf_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}
            }
        ).execute()
        
        if not result.data:
            return "No relevant content found in the documents."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
From {doc['file_name']}, Page {doc['page_number']}:
Title: {doc['title']}
Content: {doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return f"Error retrieving content: {str(e)}"

@pdf_expert.tool
async def list_pdf_documents(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available PDF documents.
    
    Returns:
        List[str]: List of unique PDF file names with their page counts
    """
    try:
        # Query Supabase for unique file names and their page counts
        result = ctx.deps.supabase.from_('pdf_pages') \
            .select('file_name, page_number') \
            .execute()
        
        if not result.data:
            return []
            
        # Process results to get unique files and their page counts
        file_stats = {}
        for doc in result.data:
            file_name = doc['file_name']
            if file_name not in file_stats:
                file_stats[file_name] = set()
            file_stats[file_name].add(doc['page_number'])
        
        # Format the results
        return [f"{file_name} ({len(pages)} pages)" for file_name, pages in sorted(file_stats.items())]
        
    except Exception as e:
        print(f"Error retrieving PDF documents: {e}")
        return []

@pdf_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], file_name: str, page_number: int) -> str:
    """
    Retrieve the full content of a specific PDF page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        file_name: The name of the PDF file
        page_number: The page number to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this page, ordered by chunk number
        result = ctx.deps.supabase.from_('pdf_pages') \
            .select('title, content, metadata') \
            .eq('file_name', file_name) \
            .eq('page_number', page_number) \
            .execute()
        
        if not result.data:
            return f"No content found for {file_name} page {page_number}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title']
        formatted_content = [f"# {page_title}\nFile: {file_name}, Page: {page_number}\n"]
        
        # Sort chunks by their chunk number from metadata
        chunks = sorted(result.data, key=lambda x: x['metadata'].get('chunk_number', 1))
        
        # Add each chunk's content
        for chunk in chunks:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

async def delete_document_records(file_name: str):
    """Delete all records for a specific document from Supabase."""
    try:
        # First, verify how many records exist for this file
        count_before = supabase.table("pdf_pages").select("id").eq("file_name", file_name).execute()
        if count_before.data:
            print(f"Found {len(count_before.data)} records to delete for {file_name}")
            
            # Delete the records
            result = supabase.table("pdf_pages").delete().eq("file_name", file_name).execute()
            
            # Verify deletion
            count_after = supabase.table("pdf_pages").select("id").eq("file_name", file_name).execute()
            if count_after.data:
                print(f"Warning: {len(count_after.data)} records still exist for {file_name}")
                # Try one more time with a small delay
                await asyncio.sleep(1)
                supabase.table("pdf_pages").delete().eq("file_name", file_name).execute()
            
            print(f"Successfully deleted {len(count_before.data)} records for {file_name}")
            return result
        else:
            print(f"No records found for {file_name}")
            return None
    except Exception as e:
        print(f"Error deleting Supabase records for {file_name}: {e}")
        raise e

async def verify_and_clean_database():
    """Verify and clean up the database by removing records for non-existent files."""
    try:
        # Get all PDF files in the filesystem
        pdf_files = {pdf_file.name for pdf_file in get_pdf_files()}
        
        # Get all unique file names from the database
        result = supabase.table("pdf_pages").select("file_name").execute()
        db_files = {record['file_name'] for record in result.data}
        
        # Find files in DB that don't exist in filesystem
        files_to_delete = db_files - pdf_files
        
        if files_to_delete:
            print(f"Found {len(files_to_delete)} files in database that don't exist in filesystem")
            for file_name in files_to_delete:
                await delete_document_records(file_name)
                print(f"Cleaned up records for missing file: {file_name}")
        
        return len(files_to_delete)
    except Exception as e:
        print(f"Error during database cleanup: {e}")
        raise e

def document_management_tab():
    """Document management interface."""
    st.header("📄 Document Management")
    
    # Create necessary directories
    os.makedirs("pdf_docs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # PDF Upload Section
    st.subheader("Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Maximum file size: 1GB per file"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024 * 1024)  # Size in GB
            if file_size > 1:
                st.error(f"File {uploaded_file.name} exceeds 1GB limit")
                continue
                
            # Save PDF file
            pdf_path = os.path.join("pdf_docs", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success(f"Uploaded: {uploaded_file.name}")
    
    # Document List and Management
    st.subheader("Manage Documents")
    pdf_files = get_pdf_files()
    
    # Add database cleanup button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🧹 Clean DB", help="Remove database records for non-existent files"):
            with st.spinner("Cleaning database..."):
                cleaned_count = asyncio.run(verify_and_clean_database())
                if cleaned_count > 0:
                    st.success(f"Cleaned up records for {cleaned_count} missing files")
                else:
                    st.success("Database is clean")
    
    if pdf_files:
        st.write(f"Found {len(pdf_files)} PDF documents:")
        
        # Create a container for the process button and progress bar
        process_container = st.container()
        with process_container:
            # Center the button using columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Process Documents", key="process_docs", type="primary", use_container_width=True):
                    process_pdfs()
        
        # List all PDF files with delete buttons
        for pdf_file in pdf_files:
            col1, col2 = st.columns([6, 1])
            with col1:
                st.write(f"📄 {pdf_file.name}")
            with col2:
                if st.button("🗑️", key=f"delete_{pdf_file.name}"):
                    try:
                        # Delete the file from the filesystem
                        os.remove(pdf_file)
                        
                        # Delete records from Supabase
                        with st.spinner(f"Deleting {pdf_file.name} and its records..."):
                            asyncio.run(delete_document_records(pdf_file.name))
                        
                        st.success(f"Deleted: {pdf_file.name} and its records")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting {pdf_file.name}: {str(e)}")
    else:
        st.info("No PDF documents found. Please upload some PDF files.")

def chat_interface_tab():
    """Chat interface for querying documents."""
    st.header("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message["role"], message["content"])
    
    # Query input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question about your documents:", height=100)
        submit_button = st.form_submit_button("Send", use_container_width=True)
        
        if submit_button and user_input:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Process query using PydanticAI agent
            with st.spinner("Processing query..."):
                try:
                    # Run the agent
                    result = asyncio.run(pdf_expert.run(user_input, deps=deps))
                    response = result.data
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Save chat history
                    with open("./data/chat_history.json", "w", encoding='utf-8') as f:
                        json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            st.rerun()

def chat_history_tab():
    """Chat history interface with collapsible answers and deletion options."""
    st.header("💭 Chat History")
    
    # Delete all button
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("🗑️ Delete All Chat History"):
            try:
                # Clear chat history from session state
                st.session_state.chat_history = []
                
                # Clear the chat history file
                with open("./data/chat_history.json", "w", encoding='utf-8') as f:
                    json.dump([], f)
                
                st.success("Chat history cleared!")
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing chat history: {str(e)}")
    
    # Display chat history after the delete button
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        # Group messages into Q&A pairs
        for i in range(0, len(st.session_state.chat_history), 2):
            if i + 1 < len(st.session_state.chat_history):
                question = st.session_state.chat_history[i]
                answer = st.session_state.chat_history[i + 1]
                
                # Create container for each Q&A pair
                qa_container = st.container()
                with qa_container:
                    # Create columns for the question and delete button
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(f"**Question:** {question['content']}")
                    with col2:
                        # Delete button for this Q&A pair
                        if st.button("🗑️", key=f"delete_{i}"):
                            # Remove this Q&A pair
                            st.session_state.chat_history.pop(i+1)  # Remove answer first
                            st.session_state.chat_history.pop(i)    # Then remove question
                            # Save updated chat history
                            with open("./data/chat_history.json", "w", encoding='utf-8') as f:
                                json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
                            st.rerun()
                    
                    # Create expandable answer section
                    with st.expander("Show Answer", expanded=False):
                        st.markdown(answer['content'])
                    
                    # Add a visual separator between Q&A pairs
                    st.markdown("---")
    else:
        st.info("No chat history available")

def main():
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 PDF Document Question Answering")
        st.markdown("---")
        st.markdown("""
        ### Features
        - Upload and manage PDF documents
        - Process documents for RAG
        - Query documents using natural language
        """)
        
        # Display document stats
        st.markdown("### Document Status")
        pdf_count = len(get_pdf_files())
        st.info(f"📄 {pdf_count} PDF documents loaded")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📑 Document Management", "💭 Chat History"])
    
    with tab1:
        chat_interface_tab()
    
    with tab2:
        document_management_tab()
        
    with tab3:
        chat_history_tab()

if __name__ == "__main__":
    main() 