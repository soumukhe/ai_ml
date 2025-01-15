import streamlit as st
import time
from main_rag import RAGSystem
import os
import subprocess
import glob
import shutil
import json
from pypdf import PdfReader
from pathlib import Path
import re
import sys

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer warnings
os.environ["ALLOW_RESET"] = "true"  # Enable ChromaDB reset functionality

# Set page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Increase maximum upload size to 1GB
st.config.set_option('server.maxUploadSize', 1024)  # Size in MB

# Initialize theme in session state if not present
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'
if 'confirm_delete' not in st.session_state:
    st.session_state.confirm_delete = False

# Apply theme-specific CSS
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        /* Base dark theme */
        .stApp { 
            background-color: #1E1E1E; 
            color: #E0E0E0; 
        }
        
        /* Sidebar */
        .css-1d391kg, .css-12oz5g7 {
            background-color: #252525 !important;
        }
        .css-1d391kg p, .css-12oz5g7 p {
            color: #E0E0E0 !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #252525 !important;
            color: #E0E0E0 !important;
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: #E0E0E0 !important;
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] h4 {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] code {
            color: #FF9D00 !important;
            background-color: #2D2D2D !important;
        }
        
        /* Buttons with better contrast */
        .stButton button { 
            background-color: #383838 !important; 
            color: #FFFFFF !important;
            border: 1px solid #505050 !important;
        }
        .stButton button:hover {
            border-color: #00A4E4 !important;
        }
        
        /* Input fields with better visibility */
        .stTextInput input, .stTextArea textarea, .stSelectbox select { 
            background-color: #2D2D2D !important; 
            color: #FFFFFF !important;
            border: 1px solid #505050 !important;
        }
        
        /* Markdown text */
        .markdown-text-container {
            color: #E0E0E0 !important;
        }
        .element-container, .stMarkdown, .stMarkdown p {
            color: #E0E0E0 !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #FFFFFF !important;
        }
        
        /* Code blocks */
        code {
            color: #FF9D00 !important;
            background-color: #2D2D2D !important;
        }
        pre {
            background-color: #2D2D2D !important;
        }
        
        /* Links */
        a {
            color: #00A4E4 !important;
        }
        a:hover {
            color: #66C4FF !important;
        }
        
        /* Expander and other interactive elements */
        .streamlit-expanderHeader {
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
        }
        
        /* Success/Info/Warning/Error messages */
        .stSuccess, .stInfo {
            background-color: rgba(40, 167, 69, 0.2) !important;
            border: 1px solid #28a745 !important;
            color: #E0E0E0 !important;
        }
        
        .stWarning {
            background-color: rgba(255, 193, 7, 0.2) !important;
            border: 1px solid #ffc107 !important;
            color: #E0E0E0 !important;
        }
        
        .stError {
            background-color: rgba(220, 53, 69, 0.2) !important;
            border: 1px solid #dc3545 !important;
            color: #E0E0E0 !important;
        }
        
        /* Tables */
        .stTable {
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
        }
        
        /* Metrics */
        .stMetric {
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
        }
        div[data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }
        div[data-testid="stMetricDelta"] {
            color: #E0E0E0 !important;
        }
        
        /* Progress bars */
        .stProgress > div > div {
            background-color: #383838 !important;
        }
        
        /* File uploader */
        .uploadedFile {
            background-color: #2D2D2D !important;
            color: #FFFFFF !important;
            border: 1px solid #505050 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #252525 !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #E0E0E0 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #FFFFFF !important;
        }
        
        /* Radio buttons and checkboxes */
        .stRadio label, .stCheckbox label {
            color: #E0E0E0 !important;
        }
        
        /* Form submit buttons (like Start Crawling) */
        .stFormSubmitButton button {
            background-color: #00A4E4 !important;
            color: white !important;
            border: 1px solid #66C4FF !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 1.1em !important;
            transition: all 0.3s ease !important;
        }
        .stFormSubmitButton button:hover {
            background-color: #0086BF !important;
            border-color: #99D6FF !important;
            box-shadow: 0 0 10px rgba(102, 196, 255, 0.3) !important;
            transform: translateY(-1px) !important;
        }
        .stFormSubmitButton button:active {
            transform: translateY(1px) !important;
        }
        
        /* Make download buttons more prominent */
        .stDownloadButton button {
            background-color: #0E86D4 !important;
            color: white !important;
            border: 1px solid #66C4FF !important;
            font-weight: 600 !important;
            padding: 0.5rem 1rem !important;
        }
        .stDownloadButton button:hover {
            background-color: #66C4FF !important;
            border-color: #0E86D4 !important;
            color: #FFFFFF !important;
        }
        
        /* Style primary buttons differently */
        .stButton button[kind="primary"] {
            background-color: #FF4B4B !important;
            color: white !important;
            border: 1px solid #FF6B6B !important;
        }
        .stButton button[kind="primary"]:hover {
            background-color: #FF6B6B !important;
            border-color: #FF4B4B !important;
        }
        
        /* Secondary buttons */
        .stButton button[kind="secondary"] {
            background-color: #383838 !important;
            color: #FFFFFF !important;
            border: 1px solid #505050 !important;
        }
        .stButton button[kind="secondary"]:hover {
            background-color: #505050 !important;
            border-color: #383838 !important;
        }
        </style>
    """, unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    /* Override file uploader text */
    .stFileUploader div[data-testid="stMarkdownContainer"] p {
        display: none;
    }
    .stFileUploader::after {
        content: "Limit 1GB per file • PDF";
        color: rgba(49, 51, 63, 0.6);
        font-size: 14px;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .chat-message.assistant {
        background-color: var(--background-color);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .source-info {
        font-size: 0.8rem;
        color: var(--text-color);
        opacity: 0.8;
        margin-top: 0.5rem;
    }
    code {
        padding: 0.2em 0.4em;
        border-radius: 3px;
        background-color: var(--secondary-background-color);
    }
    pre {
        padding: 1em;
        border-radius: 5px;
        background-color: var(--secondary-background-color);
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
        opacity: 0.9;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    # Ensure directories exist
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./chroma_db", exist_ok=True)
    
    # Initialize chat history from file if it exists
    if 'chat_history' not in st.session_state:
        try:
            if os.path.exists("./chroma_db/chat_history.json"):
                with open("./chroma_db/chat_history.json", "r", encoding='utf-8') as f:
                    st.session_state.chat_history = json.load(f)
            else:
                st.session_state.chat_history = []
                with open("./chroma_db/chat_history.json", "w", encoding='utf-8') as f:
                    json.dump([], f)
        except Exception as e:
            print(f"Error loading chat history: {e}")
            st.session_state.chat_history = []
    
    # Initialize RAG system if needed
    if 'rag_system' not in st.session_state or st.session_state.rag_system is None:
        try:
            print("Creating new RAG system instance...")
            st.session_state.rag_system = RAGSystem()
            
            # Check for existing embeddings in JSON
            if os.path.exists("./chroma_db/documents.json") and os.path.exists("./chroma_db/document_hashes.json"):
                print("Found existing embeddings. Loading from JSON...")
                st.session_state.rag_system.initialize_chroma()
                st.session_state.rag_system.load_from_json()
                print("Successfully loaded existing embeddings")
            else:
                print("No existing embeddings found. Please process documents from Document Management tab.")
                
        except Exception as e:
            error_msg = str(e)
            if "GetElementType is not implemented" in error_msg:
                st.error("ONNX runtime error detected. Please try the following steps:\n"
                        "1. Delete the contents of the chroma_db directory (except chat_history.json)\n"
                        "2. Restart the application\n"
                        "3. Process your documents again")
            else:
                st.error(f"Error initializing RAG system: {error_msg}")
            st.info("If the error persists, try clearing your browser cache and restarting the application.")
            print(f"Detailed initialization error: {error_msg}")
            st.session_state.rag_system = None
    
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Chat"

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling."""
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div><strong>{'You' if role == 'user' else '🤖 Assistant'}</strong></div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def run_crawler(sitemap_url: str, max_concurrent: int = 10, delay: float = 1.0):
    """Run the crawler with the specified sitemap URL."""
    try:
        # Get progress bar and status text from session state
        progress_bar = st.session_state.get('progress_bar')
        status_text = st.session_state.get('status_text')
        
        # Reset RAG system first
        if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
            if progress_bar and status_text:
                progress_bar.progress(20)
                status_text.text("Cleaning up previous data...")
            RAGSystem.reset_instance()  # Use the reset_instance method
            st.session_state.rag_system = None
        
        # Clear ChromaDB files (except chat history)
        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            for item in os.listdir(chroma_db_path):
                if item != "chat_history.json":
                    item_path = os.path.join(chroma_db_path, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"Error removing {item}: {e}")
        
        # Update progress
        if progress_bar and status_text:
            progress_bar.progress(30)
            status_text.text("Preparing crawler...")
        
        # Read the original crawler script
        with open('3-crawl_parallel_saveMarkdown.py', 'r') as f:
            script_content = f.read()
        
        # Update the sitemap_url in the script
        updated_script = re.sub(
            r'sitemap_url = "[^"]*"',
            f'sitemap_url = "{sitemap_url}"',
            script_content
        )
        
        # Update max_concurrent
        updated_script = re.sub(
            r'max_concurrent = \d+',
            f'max_concurrent = {max_concurrent}',
            updated_script
        )
        
        # Write the updated script back
        with open('3-crawl_parallel_saveMarkdown.py', 'w') as f:
            f.write(updated_script)
        
        # Update progress
        if progress_bar and status_text:
            progress_bar.progress(40)
            status_text.text("Starting crawler...")
        
        print(f"Starting crawler with URL: {sitemap_url}")
        
        # Run the crawler script
        process = subprocess.Popen(
            [sys.executable, '3-crawl_parallel_saveMarkdown.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Update progress
        if progress_bar and status_text:
            progress_bar.progress(50)
            status_text.text("Crawling pages...")
        
        # Wait for the process to complete
        stdout, stderr = process.communicate()
        
        # Print output for debugging
        print("Crawler output:", stdout)
        if stderr:
            print("Crawler errors:", stderr)
        
        # Update progress
        if progress_bar and status_text:
            progress_bar.progress(80)
            status_text.text("Processing results...")
        
        # Check if markdown files were created
        markdown_files = glob.glob('data/*.md')
        if markdown_files:
            print(f"Created {len(markdown_files)} markdown files")
            
            # Update progress
            if progress_bar and status_text:
                progress_bar.progress(90)
                status_text.text("Initializing RAG system...")
            
            # Initialize new RAG system
            st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.load_documents()
            st.session_state.rag_system.save_to_json()
            
            return True, "Crawling completed successfully!"
        else:
            print("No markdown files were created")
            return False, "No markdown files were created during crawling"
            
    except Exception as e:
        print(f"Error running crawler: {e}")
        return False, f"Error running crawler: {str(e)}"

def get_file_stats(file_path):
    """Get statistics about a markdown file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = content.split('\n')
            code_blocks = content.count('```')
            size = os.path.getsize(file_path) / 1024  # KB
            return {
                'lines': len(lines),
                'code_blocks': code_blocks // 2,  # Divide by 2 as each block has start and end
                'size': f"{size:.1f} KB",
                'last_modified': time.ctime(os.path.getmtime(file_path))
            }
    except Exception:
        return None

def analyze_content(file_path):
    """Analyze markdown file content."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # Basic analysis
            total_chars = len(content)
            words = len(content.split())
            code_blocks = content.count('```') // 2
            links = content.count('](')
            headings = sum(1 for line in content.split('\n') if line.strip().startswith('#'))
            
            # Code language detection
            code_sections = content.split('```')[1::2]  # Get content between ``` markers
            languages = {}
            for section in code_sections:
                lang = section.split('\n')[0].strip()
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1
            
            return {
                'total_chars': total_chars,
                'words': words,
                'code_blocks': code_blocks,
                'links': links,
                'headings': headings,
                'languages': languages
            }
    except Exception:
        return None

def export_files(files, export_dir="exports"):
    """Export selected files to a zip archive."""
    import zipfile
    import datetime
    
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(export_dir, f"markdown_export_{timestamp}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, os.path.basename(file))
    
    return zip_path

def document_management_tab():
    """Document management tab for uploading PDFs and managing files."""
    st.header("Document Management")
    
    # Load documents button
    if st.button("Load Documents"):
        with st.spinner("Loading documents..."):
            try:
                # This will now only load if documents have changed
                rag.load_documents()
                st.success("Documents loaded successfully!")
            except Exception as e:
                st.error(f"Error loading documents: {str(e)}")
                st.stop()
    
    # PDF Upload Section
    st.header("PDF Document Upload")
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("pdf_docs", exist_ok=True)
    
    # Create a container for the file uploader
    upload_container = st.container()
    with upload_container:
        # File uploader with 1GB limit
        uploaded_pdf = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Maximum file size: 1GB"
        )
    
    if uploaded_pdf is not None:
        file_size = len(uploaded_pdf.getvalue()) / (1024 * 1024 * 1024)  # Size in GB
        if file_size > 1:
            st.error("File size exceeds 1GB limit. Please upload a smaller file.")
        else:
            # Save PDF to pdf_docs directory
            pdf_path = os.path.join("pdf_docs", uploaded_pdf.name)
            
            with st.spinner("Processing PDF..."):
                try:
                    # Save the uploaded PDF
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_pdf.getvalue())
                    
                    # Convert to markdown
                    markdown_path = convert_pdf_to_markdown(pdf_path)
                    st.success(f"PDF successfully converted to markdown: {os.path.basename(markdown_path)}")
                    
                    # Process the new document
                    with st.spinner("Processing and embedding document..."):
                        try:
                            # Reset any existing RAG system
                            RAGSystem.reset_instance()
                            
                            # Clear ChromaDB files (except chat history)
                            chroma_db_path = "./chroma_db"
                            if os.path.exists(chroma_db_path):
                                for item in os.listdir(chroma_db_path):
                                    if item != "chat_history.json":
                                        item_path = os.path.join(chroma_db_path, item)
                                        try:
                                            if os.path.isfile(item_path):
                                                os.unlink(item_path)
                                            elif os.path.isdir(item_path):
                                                shutil.rmtree(item_path)
                                        except Exception as e:
                                            print(f"Error removing {item}: {e}")
                            
                            # Initialize new RAG system
                            print("Creating new RAG system instance...")
                            st.session_state.rag_system = RAGSystem()
                            
                            # Process documents and save to JSON
                            print("Loading documents...")
                            st.session_state.rag_system.load_documents()
                            
                            # Verify documents were processed
                            try:
                                doc_count = len(st.session_state.rag_system.collection.get()["documents"])
                                if doc_count > 0:
                                    print(f"Saving {doc_count} documents to JSON...")
                                    st.session_state.rag_system.save_to_json()
                                    st.success(f"Document processed successfully! Found {doc_count} chunks.")
                                else:
                                    st.error("No documents were processed. Please try again.")
                                    if os.path.exists(pdf_path):
                                        os.remove(pdf_path)
                                    if os.path.exists(markdown_path):
                                        os.remove(markdown_path)
                            except Exception as e:
                                print(f"Error verifying document processing: {e}")
                                st.error(f"Error verifying document processing: {str(e)}")
                                st.info("Please try uploading the document again.")
                                # Clean up on error
                                if os.path.exists(pdf_path):
                                    os.remove(pdf_path)
                                if os.path.exists(markdown_path):
                                    os.remove(markdown_path)
                        except Exception as e:
                            print(f"Error processing document: {e}")
                            st.error(f"Error processing document: {str(e)}")
                            st.info("Try restarting the application if the error persists.")
                            # Clean up on error
                            if os.path.exists(pdf_path):
                                os.remove(pdf_path)
                            if os.path.exists(markdown_path):
                                os.remove(markdown_path)
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    # Clean up on error
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
    
    # Crawling Section
    st.header("Web Crawling")
    
    # Create placeholders for progress tracking
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    with st.form("crawler_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            sitemap_url = st.text_input(
                "Enter Website URL to Crawl:",
                placeholder="https://example.com/sitemap.xml",
                help="Enter the sitemap URL or website URL to crawl"
            )
        with col2:
            st.markdown("### Advanced Settings")
            max_concurrent = st.number_input("Max Concurrent", 
                                          min_value=1, max_value=10, 
                                          value=5,
                                          help="Number of parallel crawling processes")
            delay = st.number_input("Delay (seconds)", 
                                  min_value=0.0, max_value=5.0, 
                                  value=1.0, step=0.1,
                                  help="Delay between requests")
        
        crawl_button = st.form_submit_button("Start Crawling")
        
    if crawl_button and sitemap_url:
        try:
            # Initialize progress bar and status
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.text("Initializing crawler...")
            
            # Store in session state
            st.session_state['progress_bar'] = progress_bar
            st.session_state['status_text'] = status_text
            
            # Update initial progress
            progress_bar.progress(10)
            
            # Run the crawler
            success, message = run_crawler(sitemap_url, max_concurrent, delay)
            
            if success:
                # Update progress for successful crawl
                progress_bar.progress(100)
                status_text.text("Crawling completed successfully!")
                st.success(message)
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(message)
                
        except Exception as e:
            if 'progress_bar' in st.session_state:
                st.session_state.progress_bar.empty()
            if 'status_text' in st.session_state:
                st.session_state.status_text.empty()
            st.error(f"Error during crawling: {str(e)}")
            
        finally:
            # Clean up progress indicators after a delay
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            if 'progress_bar' in st.session_state:
                del st.session_state['progress_bar']
            if 'status_text' in st.session_state:
                del st.session_state['status_text']
    
    # Document Management Section
    st.header("Document Management")
    
    # Get list of markdown files
    markdown_files = glob.glob("data/*.md")
    
    if not markdown_files:
        st.warning("No markdown files found in the data directory.")
    else:
        # Search and Filter Section
        st.subheader("Search and Filter")
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            search_term = st.text_input("Search in files:", 
                                      help="Search in file names and content")
        with col2:
            content_filter = st.multiselect("Filter by content:",
                                          ["Has Code", "Has Links", "Has Headings"])
        with col3:
            min_size = st.number_input("Min Size (KB)", min_value=0, value=0)
        
        # Sort and Display Options
        sort_col1, sort_col2, sort_col3 = st.columns([2, 2, 1])
        with sort_col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Name", "Size", "Last Modified", "Lines Count", "Code Blocks"]
            )
        with sort_col2:
            show_stats = st.checkbox("Show detailed statistics", value=True)
        with sort_col3:
            show_analysis = st.checkbox("Show content analysis", value=False)
        
        # Batch Operations
        st.subheader("Batch Operations")
        batch_col1, batch_col2, batch_col3 = st.columns([1, 1, 1])
        with batch_col1:
            if st.button("Export All Files"):
                zip_path = export_files(markdown_files)
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        "Download ZIP",
                        f,
                        file_name=os.path.basename(zip_path),
                        mime="application/zip"
                    )
        
        with batch_col2:
            delete_col1, delete_col2 = st.columns([3, 2])
            with delete_col1:
                if st.button("Delete All Files", type="secondary", key="delete_all"):
                    st.session_state.confirm_delete = True
            
            if st.session_state.confirm_delete:
                with delete_col2:
                    st.checkbox("Confirm deletion", key="confirm_checkbox", 
                              help="Check this box to confirm deletion of all files")
                if st.session_state.get("confirm_checkbox", False):
                    try:
                        # Stop the crawler process if it's running
                        os.system("pkill -f '3-crawl_parallel_saveMarkdown.py'")
                        
                        # First, stop any running RAG system
                        RAGSystem.reset_instance()
                        st.session_state.rag_system = None
                        
                        # Delete all markdown files from data directory
                        data_dir = "./data"
                        if os.path.exists(data_dir):
                            markdown_files = glob.glob(os.path.join(data_dir, "*.md"))
                            for file_path in markdown_files:
                                try:
                                    os.remove(file_path)
                                    st.success(f"Deleted {os.path.basename(file_path)}")
                                except Exception as e:
                                    st.error(f"Error deleting {os.path.basename(file_path)}: {str(e)}")
                        
                        # Delete PDF files from pdf_docs directory
                        pdf_dir = "./pdf_docs"
                        if os.path.exists(pdf_dir):
                            pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
                            for pdf_path in pdf_files:
                                try:
                                    os.remove(pdf_path)
                                    st.success(f"Deleted PDF: {os.path.basename(pdf_path)}")
                                except Exception as e:
                                    st.error(f"Error deleting PDF {os.path.basename(pdf_path)}: {str(e)}")
                        
                        # Clear ChromaDB files (except chat history)
                        chroma_db_path = "./chroma_db"
                        if os.path.exists(chroma_db_path):
                            for item in os.listdir(chroma_db_path):
                                if item != "chat_history.json":
                                    item_path = os.path.join(chroma_db_path, item)
                                    try:
                                        if os.path.isfile(item_path):
                                            os.unlink(item_path)
                                        elif os.path.isdir(item_path):
                                            shutil.rmtree(item_path)
                                    except Exception as e:
                                        st.error(f"Error removing {item}: {str(e)}")
                        
                        # Reset the crawler script
                        with open('3-crawl_parallel_saveMarkdown.py', 'r') as file:
                            content = file.read()
                        
                        # Reset the configuration to defaults and clear any running state
                        modified_content = content.replace(
                            'sitemap_url = "' + content.split('sitemap_url = "')[1].split('"')[0] + '"',
                            'sitemap_url = ""'  # Set to empty to prevent auto-crawling
                        )
                        
                        with open('3-crawl_parallel_saveMarkdown.py', 'w') as file:
                            file.write(modified_content)
                        
                        st.session_state.confirm_delete = False
                        st.success("All files deleted successfully!")
                        time.sleep(1)  # Give time for the messages to be shown
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during deletion: {str(e)}")
        
        with batch_col3:
            uploaded_file = st.file_uploader("Import Markdown", type=['md'])
            if uploaded_file:
                try:
                    with open(os.path.join("data", uploaded_file.name), 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    st.success(f"Imported {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error importing file: {str(e)}")
        
        # Filter files based on search and filters
        filtered_files = markdown_files.copy()
        if search_term:
            filtered_files = [f for f in filtered_files if search_term.lower() in 
                            os.path.basename(f).lower() or 
                            search_term.lower() in open(f, 'r').read().lower()]
        
        if content_filter:
            file_analyses = {f: analyze_content(f) for f in filtered_files}
            if "Has Code" in content_filter:
                filtered_files = [f for f in filtered_files if file_analyses[f]['code_blocks'] > 0]
            if "Has Links" in content_filter:
                filtered_files = [f for f in filtered_files if file_analyses[f]['links'] > 0]
            if "Has Headings" in content_filter:
                filtered_files = [f for f in filtered_files if file_analyses[f]['headings'] > 0]
        
        if min_size > 0:
            filtered_files = [f for f in filtered_files if os.path.getsize(f) / 1024 >= min_size]
        
        # Sort files
        if sort_by == "Name":
            filtered_files.sort()
        elif sort_by in ["Size", "Last Modified", "Lines Count"]:
            file_stats = {f: get_file_stats(f) for f in filtered_files}
            if sort_by == "Size":
                filtered_files.sort(key=lambda x: float(file_stats[x]['size'].split()[0]) if file_stats[x] else 0, reverse=True)
            elif sort_by == "Last Modified":
                filtered_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            elif sort_by == "Lines Count":
                filtered_files.sort(key=lambda x: file_stats[x]['lines'] if file_stats[x] else 0, reverse=True)
        
        st.write(f"Found {len(filtered_files)} matching files out of {len(markdown_files)} total:")
        
        # Display files
        for file_path in filtered_files:
            filename = os.path.basename(file_path)
            stats = get_file_stats(file_path) if show_stats else None
            analysis = analyze_content(file_path) if show_analysis else None
            
            with st.expander(f"📄 {filename}", expanded=False):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    if stats:
                        st.markdown(f"""
                        - **Size**: {stats['size']}
                        - **Lines**: {stats['lines']}
                        - **Code Blocks**: {stats['code_blocks']}
                        - **Last Modified**: {stats['last_modified']}
                        """)
                    
                    if analysis:
                        st.markdown("#### Content Analysis")
                        st.markdown(f"""
                        - **Words**: {analysis['words']}
                        - **Characters**: {analysis['total_chars']}
                        - **Links**: {analysis['links']}
                        - **Headings**: {analysis['headings']}
                        - **Programming Languages**: {', '.join(analysis['languages'].keys()) if analysis['languages'] else 'None'}
                        """)
                
                with col2:
                    if st.button(f"View Content 👁️", key=f"view_{filename}"):
                        with st.spinner("Loading file content..."):
                            with open(file_path, 'r') as f:
                                content = f.read()
                            st.code(content, language='markdown')
                    
                    if st.button(f"Export File 📤", key=f"export_{filename}"):
                        with open(file_path, 'r') as f:
                            content = f.read()
                            st.download_button(
                                "Download File",
                                content,
                                file_name=filename,
                                mime="text/markdown"
                            )
                
                with col3:
                    if st.button(f"Delete File 🗑️", key=f"delete_{filename}"):
                        try:
                            os.remove(file_path)
                            st.success(f"Deleted {filename}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {filename}: {str(e)}")
    
    # Reprocess Documents Section
    st.header("Reprocess Documents")
    col1, col2 = st.columns([2, 2])
    with col1:
        chunk_size = st.number_input("Chunk Size", 
                                   min_value=500, max_value=3000, 
                                   value=1500, step=100,
                                   help="Size of text chunks for processing")
    
    if st.button("Reprocess All Documents", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Clean up existing RAG system
            status_text.text("Preparing for reprocessing...")
            progress_bar.progress(10)
            
            # Clean up existing instance
            if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system is not None:
                if hasattr(st.session_state.rag_system, 'client'):
                    if hasattr(st.session_state.rag_system.client, '_client'):
                        st.session_state.rag_system.client._client.close()
                    st.session_state.rag_system.client = None
                st.session_state.rag_system = None
            
            # Remove ChromaDB files
            chroma_db_path = "./chroma_db"
            if os.path.exists(chroma_db_path):
                for item in os.listdir(chroma_db_path):
                    if item != "chat_history.json":
                        item_path = os.path.join(chroma_db_path, item)
                        try:
                            if os.path.isfile(item_path):
                                os.unlink(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                        except Exception as e:
                            print(f"Error removing {item_path}: {e}")
            
            time.sleep(1)  # Give time for cleanup
            
            # Initialize new RAG system
            status_text.text("Initializing new RAG system...")
            progress_bar.progress(30)
            st.session_state.rag_system = RAGSystem()
            
            # Process all documents
            status_text.text("Processing documents...")
            progress_bar.progress(60)
            markdown_files = glob.glob("data/*.md")
            if not markdown_files:
                raise Exception("No markdown files found to process")
                
            st.session_state.rag_system.load_documents()
            
            # Save to JSON for persistence
            status_text.text("Saving embeddings...")
            progress_bar.progress(90)
            st.session_state.rag_system.save_to_json()
            
            status_text.text("Documents reprocessed successfully!")
            progress_bar.progress(100)
            st.success("Documents reprocessed and saved successfully!")
            
        except Exception as e:
            st.error(f"Error reprocessing documents: {str(e)}")
            st.info("Try restarting the application if the error persists.")
            print(f"Detailed error: {str(e)}")  # For debugging
        finally:
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()

def verify_rag_system():
    """Verify RAG system is properly initialized and ready."""
    try:
        # Check if RAG system exists and is properly initialized
        if (hasattr(st.session_state, 'rag_system') and 
            st.session_state.rag_system is not None and 
            hasattr(st.session_state.rag_system, 'collection')):
            try:
                # Quick verification of collection
                count = len(st.session_state.rag_system.collection.get()["documents"])
                if count > 0:
                    print("RAG system already initialized with documents")
                    return True, None
            except Exception as e:
                print(f"Error verifying existing collection: {e}")
        
        print("Checking for existing embeddings...")
        
        # Check for existing embeddings in JSON
        if os.path.exists("./chroma_db/documents.json") and os.path.exists("./chroma_db/document_hashes.json"):
            print("Found existing embeddings. Loading from JSON...")
            if not hasattr(st.session_state, 'rag_system') or st.session_state.rag_system is None:
                st.session_state.rag_system = RAGSystem()
            st.session_state.rag_system.initialize_chroma()
            st.session_state.rag_system.load_from_json()
            print("Successfully loaded existing embeddings")
            return True, None
        
        # If no existing embeddings, check for markdown files
        print("No existing embeddings found. Checking for documents...")
        markdown_files = glob.glob("data/*.md")
        if not markdown_files:
            return False, "No markdown files found in data directory. Please add some documents first."
        
        return False, "Documents found but not processed. Please process documents from Document Management tab."
            
    except Exception as e:
        print(f"Error in verify_rag_system: {e}")
        error_msg = str(e)
        if "GetElementType is not implemented" in error_msg:
            return False, ("ONNX runtime error detected. Please try:\n"
                         "1. Delete contents of chroma_db directory (except chat_history.json)\n"
                         "2. Restart the application\n"
                         "3. Process your documents again")
        return False, f"Error initializing RAG system: {str(e)}"

def chat_interface_tab():
    """Chat interface."""
    st.title("💬 Chat Interface")
    
    # Verify RAG system
    is_ready, error_message = verify_rag_system()
    if not is_ready:
        st.info(error_message)
        return
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
    
    # Input area
    st.markdown("---")
    with st.container():
        # Use a form to handle input properly
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area("Your question:", key="question_input", height=100)
            submit_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
            
            if submit_button and question:
                try:
                    # Verify RAG system is ready
                    if not hasattr(st.session_state.rag_system, 'client') or st.session_state.rag_system.client is None:
                        raise Exception("RAG system not properly initialized")
                    
                    # Add user message to chat
                    st.session_state.chat_history.append({"role": "user", "content": question})
                    
                    # Get response
                    with st.spinner('Thinking...'):
                        response = st.session_state.rag_system.query(question)
                    
                    # Add assistant response to chat
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Save chat history to file
                    try:
                        with open("./chroma_db/chat_history.json", "w", encoding='utf-8') as f:
                            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        st.warning(f"Failed to save chat history: {e}")
                    
                    # Rerun to update chat display
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
                    st.info("Try reprocessing documents if the error persists.")
                    print(f"Detailed error: {str(e)}")  # For debugging
                    
                    # Attempt recovery with proper initialization sequence
                    try:
                        st.session_state.rag_system = RAGSystem()
                        st.session_state.rag_system.initialize_chroma()
                        if os.path.exists("./chroma_db/documents.json"):
                            st.session_state.rag_system.load_from_json()
                            if hasattr(st.session_state.rag_system, 'collection'):
                                count = st.session_state.rag_system.collection.count()
                                if count > 0:
                                    st.warning("System recovered. Please try your question again.")
                                else:
                                    st.warning("Recovery failed - no documents found. Please reprocess documents.")
                    except Exception as recovery_error:
                        print(f"Recovery failed: {recovery_error}")
                        st.error("Recovery failed. Please try reprocessing documents.")

def toggle_delete_confirmation():
    st.session_state.confirm_delete = not st.session_state.confirm_delete

def chat_history_tab():
    """Chat history interface."""
    st.title("💭 Chat History")
    
    # Delete all button
    col1, col2 = st.columns([3,1])
    with col1:
        if st.button("🗑️ Delete All Chat History"):
            try:
                # Clear chat history from session state
                st.session_state.chat_history = []
                
                # Clear the chat history file
                with open("./chroma_db/chat_history.json", "w", encoding='utf-8') as f:
                    json.dump([], f)
                
                # Properly reinitialize RAG system with existing embeddings
                if hasattr(st.session_state, 'rag_system'):
                    try:
                        st.session_state.rag_system = RAGSystem()
                        st.session_state.rag_system.initialize_chroma()
                        if os.path.exists("./chroma_db/documents.json"):
                            st.session_state.rag_system.load_from_json()
                            if hasattr(st.session_state.rag_system, 'collection'):
                                count = st.session_state.rag_system.collection.count()
                                print(f"Reinitialized with {count} documents")
                    except Exception as e:
                        print(f"Error reinitializing RAG system: {e}")
                        st.session_state.rag_system = None
                
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
                            with open("./chroma_db/chat_history.json", "w", encoding='utf-8') as f:
                                json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
                            st.rerun()
                    
                    # Create expandable answer section
                    with st.expander("Show Answer", expanded=False):
                        st.markdown(answer['content'])
                    
                    # Add a visual separator between Q&A pairs
                    st.markdown("---")
    else:
        st.info("No chat history available")

def convert_pdf_to_markdown(pdf_path):
    """Convert a PDF file to markdown format using pypdf."""
    try:
        # Create output filename
        pdf_name = Path(pdf_path).stem
        output_path = f"data/{pdf_name}.md"
        
        # Read PDF
        reader = PdfReader(pdf_path)
        
        # Convert to markdown
        markdown_content = []
        
        # Add title
        markdown_content.append(f"# {pdf_name}\n\n")
        
        # Process each page
        for page_num, page in enumerate(reader.pages, 1):
            # Extract text
            text = page.extract_text()
            
            # Add page header
            markdown_content.append(f"## Page {page_num}\n\n")
            
            # Add page content with proper line breaks
            paragraphs = text.split('\n\n')
            for para in paragraphs:
                if para.strip():
                    markdown_content.append(f"{para.strip()}\n\n")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(markdown_content))
        
        return output_path
    except Exception as e:
        raise Exception(f"Error converting PDF to markdown: {str(e)}")

def web_crawling_tab():
    """Web Crawling tab for crawling documentation websites."""
    st.header("Web Crawling")
    
    # Input for sitemap URL
    sitemap_url = st.text_input(
        "Enter Website URL",
        help="Enter a website URL to crawl. Can be a sitemap.xml URL or a regular website URL."
    )
    
    # Create placeholder for progress bar and status
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Start crawling button
    if st.button("Start Crawling", key="start_crawling"):
        if not sitemap_url:
            st.error("Please enter a URL to crawl")
            return
        
        try:
            # Initialize progress bar and status
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.text("Initializing crawler...")
            
            # Store in session state
            st.session_state['progress_bar'] = progress_bar
            st.session_state['status_text'] = status_text
            
            # Update initial progress
            progress_bar.progress(10)
            
            # Run the crawler
            success, message = run_crawler(sitemap_url)
            
            if success:
                # Update progress for successful crawl
                progress_bar.progress(100)
                status_text.text("Crawling completed successfully!")
                st.success(message)
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(message)
                
        except Exception as e:
            if 'progress_bar' in st.session_state:
                st.session_state.progress_bar.empty()
            if 'status_text' in st.session_state:
                st.session_state.status_text.empty()
            st.error(f"Error during crawling: {str(e)}")
            
        finally:
            # Clean up progress indicators after a delay
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()
            if 'progress_bar' in st.session_state:
                del st.session_state['progress_bar']
            if 'status_text' in st.session_state:
                del st.session_state['status_text']

def main():
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG system without verifying
    try:
        rag = RAGSystem()
        # Only verify if documents.json exists
        if os.path.exists("./chroma_db/documents.json"):
            rag.verify_rag_system()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        # Add dark mode toggle at the top
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("📚 Document Q&A System")
        with col2:
            if st.button("🌓 Theme"):
                st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
                st.rerun()
        
        st.markdown("---")
        st.markdown("""
        ### About
        This system allows you to ask questions about your documents using advanced AI. 
        The system will:
        - Search through your documents
        - Find relevant information
        - Provide detailed answers with source references
        """)
        
        st.markdown("---")
        st.markdown("""
        ### Data Source
        This system supports two ways of loading documents:

        1. **Web Crawling**:
        - Parallel crawler with up to 10 concurrent processes
        - Configurable delay between requests (0-5s)
        - Supports both sitemap URLs and direct webpage URLs
        - Default source: `ai.pydantic.dev/sitemap.xml`
        
        2. **PDF Documents**:
        - Direct upload through Document Management tab
        - Support for PDF files up to 1GB
        - Automatic conversion to markdown format
        - Preserves document structure and formatting
        
        **Document Processing Pipeline**:
        1. **Document Ingestion**:
           - Web crawling with JavaScript support
           - PDF to markdown conversion
           - Structure preservation
           - Smart formatting
        
        2. **Text Processing**:
           - Chunk size: 1500 tokens (configurable 500-3000)
           - Smart chunking preserves document structure
           - Metadata tracking for source attribution
           - Automatic change detection
        
        3. **Vector Storage**:
           - **Embedding Model**: OpenAI's `text-embedding-3-small`
             - 1536-dimensional embeddings
             - Optimized for code and technical content
           - **Vector DB**: ChromaDB
             - Persistent storage in `./chroma_db`
             - Collection-based organization
             - Real-time similarity search
             - Automatic reindexing on changes
        
        4. **Query Processing**:
           - Semantic similarity search
           - Multi-document context assembly
           - Source attribution and tracking
           - Conversation history management
        """)
        
        st.markdown("---")
        st.markdown("### Document Status")
        try:
            doc_count = len([f for f in os.listdir("data") if f.endswith('.md')])
            st.info(f"📂 {doc_count} documents loaded and ready for queries")
        except Exception:
            st.info("Documents are loaded and ready for queries")
    
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