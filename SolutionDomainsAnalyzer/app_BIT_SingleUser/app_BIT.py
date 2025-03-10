import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import os
import re
from tqdm import tqdm
import time
import shutil
import io
import warnings
import sys
import contextlib
from langchain_experimental.utilities.python import PythonREPL
from langchain.tools import Tool
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
import json
import base64
import requests
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
import traceback
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import markdown

# Force reload of environment variables
load_dotenv(override=True)

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
        'max_retries': 7,
        'sleep_seconds': 4,
    },
    
    # Duplicate detection settings
    'duplicates': {
        'similarity_threshold': 0.95,
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

# Get environment variables without defaults
app_key = os.getenv('app_key')  # app key is required
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')


# Validate required environment variables
if not all([app_key, client_id, client_secret, langsmith_api_key]):
    raise ValueError("Missing required environment variables. Please check your .env file contains: app_key, client_id, client_secret and LANGSMITH_API_KEY")

# Initialize OpenAI client for BridgeIT
def init_azure_openai():
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
        temperature=CONFIG['agent']['temperature'],
        max_tokens=CONFIG['agent']['max_tokens'],
        model="gpt-4o",
        model_kwargs={
            "user": f'{{"appkey": "{app_key}"}}'
        }
    )
    return llm

# Fuzzy search related functions
def parse_markdown_table(markdown_text):
    """Parse markdown table into a pandas DataFrame"""
    try:
        # Check for no results message
        if "No matching results found" in markdown_text:
            return None, None

        # Split into lines and clean up
        lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
        
        if not lines or len(lines) < 3:  # Need at least header, separator, and one data row
            return None, None
            
        # Extract headers - remove leading/trailing |
        headers = [col.strip() for col in lines[0].strip('|').split('|')]
        
        # Skip separator line
        data = []
        for line in lines[2:]:  # Skip header and separator line
            if '|' in line:
                # Remove leading/trailing | and split
                row = [col.strip() for col in line.strip('|').split('|')]
                if len(row) == len(headers):  # Only add rows that match header length
                    data.append(row)
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data, columns=headers)
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    pass
            return df, None
        return None, None
    except Exception as e:
        print(f"Parse error: {str(e)}")
        return None, None

def display_fuzzy_search_results(df_result):
    """Display fuzzy search results with download options"""
    if df_result is None or len(df_result) == 0:
        st.warning("No results found for your query")
        return
        
    st.success(f"Found {len(df_result)} matching records")
    st.dataframe(df_result, use_container_width=True)
    
    # Add download buttons
    col1, col2 = st.columns(2)
    with col1:
        # Excel download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False)
        st.download_button(
            label="📥 Download as Excel",
            data=buffer.getvalue(),
            file_name="search_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # CSV download
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="search_results.csv",
            mime="text/csv"
        )

# Suppress specific PyTorch warning
warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')
warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')

# Set page config for a wider layout and professional title
st.set_page_config(
    page_title="Solutions Domain Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Initialize session state for fuzzy search
if 'fuzzy_search_df' not in st.session_state:
    st.session_state.fuzzy_search_df = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Custom CSS for a more professional look
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        margin-top: 1em;
    }
    .uploadedFile {
        border: 1px solid #ccc;
        padding: 1em;
        border-radius: 5px;
        margin: 1em 0;
    }
    .success-message {
        padding: 1em;
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 5px;
        margin: 1em 0;
    }
    /* Custom styling for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        padding: 0.5rem 1rem;
        background: #f8f9fa;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #495057;
        padding: 0.5rem 1rem;
        transition: color 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0d6efd;
    }
    .stTabs [aria-selected="true"] {
        color: #0d6efd !important;
        border-bottom: 3px solid #0d6efd !important;
        border-radius: 0;
        background: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = None

# Initialize session state for active tab
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Title and description
st.title("Solutions Domain Analyzer 🔍")
st.markdown("### Analyze and process solutions domain data with advanced NLP")

# File upload section
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (up to 1GB)", 
    type=['xlsx'], 
    help="Drag and drop or click to upload",
    accept_multiple_files=False
)

# Add Models Information section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("🤖 Models Used")

# Local Models
st.sidebar.subheader("Local Models")
with st.sidebar.expander("Sentiment Analysis", expanded=False):
    st.markdown("""
    - **Model**: facebook/bart-large-mnli
    - **Type**: Zero-shot classifier
    - **Usage**: Sentiment & importance rating
    - ✅ Runs completely offline
    """)
    
with st.sidebar.expander("Text Embeddings", expanded=False):
    st.markdown("""
    - **Model**: all-MiniLM-L6-v2
    - **Type**: SentenceTransformer
    - **Usage**: Duplicate detection
    - ✅ Fully offline processing
    - 384-dimensional embeddings
    """)

# Cloud Component
st.sidebar.subheader("Secure Search")
with st.sidebar.expander("Fuzzy Search", expanded=False):
    st.markdown("""
    - **Framework**: Langchain
    - **Model**: BridgeIT (Cisco Internal)
    - ✅ Secure enterprise authentication
    - ✅ Enterprise-grade security
    - ✅ Advanced natural language processing
    """)

st.sidebar.markdown("---")

def clean_data_directory():
    """Remove existing Excel files from data directory"""
    if not os.path.exists('data'):
        os.makedirs('data')
    for file in os.listdir('data'):
        if file.endswith('.xlsx'):
            os.remove(os.path.join('data', file))

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    clean_data_directory()
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def clean_text(text):
    """Clean and validate text for API submission"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s.,!?-]', ' ', str(text))
    text = ' '.join(text.split())
    return text

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    # Save current stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Redirect stdout/stderr to devnull
    devnull = open(os.devnull, 'w')
    sys.stdout = devnull
    sys.stderr = devnull
    
    try:
        yield
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        devnull.close()

@st.cache_resource
def load_models():
    """Load and cache the ML models"""
    try:
        # First try CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "NVIDIA GPU (CUDA)"
        # Then try MPS (for Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = "Apple M-series GPU (MPS)"
        # Fallback to CPU
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        
        st.sidebar.info(f"Using device: {device_name}")
        
        # Load embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_model.to(device)  # Move model to appropriate device
        
        # Load classifier - Note: Some models might not support MPS yet
        if device.type == "mps":
            classifier_device = -1  # Fall back to CPU for classifier if using MPS
        else:
            classifier_device = 0 if torch.cuda.is_available() else -1
            
        classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            clean_up_tokenization_spaces=True,
                            multi_label=True,
                            device=classifier_device)
        return embedding_model, classifier
    except Exception as e:
        st.warning(f"Device initialization error: {str(e)}. Falling back to CPU.")
        # Fallback to CPU
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embedding_model.to('cpu')
        classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            clean_up_tokenization_spaces=True,
                            multi_label=True,
                            device=-1)
        return embedding_model, classifier

def get_embeddings_batch(texts, model, batch_size=None):
    """Get embeddings for a batch of texts using Hugging Face model"""
    total_expected = len(texts)
    
    # Create progress tracking
    embedding_progress = st.progress(0)
    embedding_status = st.empty()
    embedding_status.text("Preparing texts for embedding...")
    
    # Clean all texts first
    cleaned_texts = [clean_text(text) for text in texts]
    
    try:
        # Determine optimal batch size based on available memory and device
        if model.device.type == 'cuda':
            # Larger batch size for GPU
            batch_size = min(CONFIG['batch']['gpu_batch_size'], len(cleaned_texts))
        else:
            # Smaller batch size for CPU/MPS
            batch_size = min(CONFIG['batch']['cpu_batch_size'], len(cleaned_texts))
        
        # Custom progress callback
        def progress_callback(current, total):
            progress = float(current) / float(total)
            embedding_progress.progress(progress)
            embedding_status.text(f"Generating embeddings... {current}/{total} texts processed")
        
        # Process all texts at once with batching
        embeddings = model.encode(
            cleaned_texts,
            batch_size=batch_size,
            show_progress_bar=False,  # We'll use our own progress bar
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=model.device,  # Use the same device as the model
            callback_steps=max(1, len(cleaned_texts) // 20),  # Update progress every 5%
            callback=progress_callback
        )
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        # Clear progress indicators
        embedding_progress.empty()
        embedding_status.empty()
        
        # Free up memory based on device type
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return embeddings
        
    except Exception as e:
        # Clear progress indicators on error
        embedding_progress.empty()
        embedding_status.empty()
        st.error(f"Error in embedding generation: {str(e)}")
        # Return zero vectors as fallback
        return np.zeros((total_expected, model.get_sentence_embedding_dimension()))

def process_domain_data(df, domain, embedding_model, classifier, skip_duplicates=False):
    """Process data for a specific domain"""
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Update progress
        status_text.text("Filtering data for selected domain...")
        progress_bar.progress(10)
        
        # Filter rows for selected Solution Domain and create an explicit copy
        filtered_domain_df = df[df['Solution Domain'] == domain].copy()
        
        # Add original row numbers (add 2 because Excel is 1-indexed and has header row)
        filtered_domain_df['Original_Row_Number'] = filtered_domain_df.index + 2
        
        # Update progress
        status_text.text("Preparing text data...")
        progress_bar.progress(20)
        
        # Create new column combining Reason and Additional Details
        filtered_domain_df.loc[:, 'Reason_W_AddDetails'] = (
            filtered_domain_df['Reason'].astype(str) + ' :: ' + 
            filtered_domain_df['Additional Details'].astype(str)
        )
        
        # Drop the original columns
        filtered_domain_df = filtered_domain_df.drop(['Reason', 'Additional Details'], axis=1)
        
        # Clean text
        status_text.text("Cleaning text data...")
        progress_bar.progress(30)
        filtered_domain_df['Reason_W_AddDetails'] = filtered_domain_df['Reason_W_AddDetails'].apply(clean_text)
        
        if not skip_duplicates:
            # Generate embeddings
            status_text.text("Generating text embeddings...")
            progress_bar.progress(40)
            texts = filtered_domain_df['Reason_W_AddDetails'].tolist()
            normalized_embeddings = get_embeddings_batch(texts, embedding_model)
            
            # Compute similarity matrix
            status_text.text("Computing similarity matrix...")
            progress_bar.progress(60)
            similarity_matrix = cosine_similarity(normalized_embeddings)
            
            # Find duplicates
            status_text.text("Detecting possible duplicates...")
            progress_bar.progress(70)
            threshold = CONFIG['duplicates']['similarity_threshold']
            possible_duplicates = [""] * len(texts)
            
            # Create a mapping of filtered index to original row number
            row_number_map = dict(enumerate(filtered_domain_df['Original_Row_Number']))
            
            for i in range(len(texts)):
                for j in range(i):
                    if similarity_matrix[i, j] > threshold:
                        original_row = row_number_map[j]
                        possible_duplicates[i] = f"Duplicate of Excel Row {original_row}"
                        break
            
            # Add duplicates column
            filtered_domain_df['possibleDuplicates'] = possible_duplicates
        
        # Process sentiments
        status_text.text("Analyzing text sentiment and request importance...")
        progress_bar.progress(80)
        sentiment_labels = CONFIG['sentiment']['labels']
        sentiments = []
        
        # Process sentiments with incremental progress
        total_texts = len(filtered_domain_df['Reason_W_AddDetails'])
        for idx, text in enumerate(filtered_domain_df['Reason_W_AddDetails']):
            try:
                result = classifier(text, 
                                 candidate_labels=sentiment_labels,
                                 hypothesis_template=CONFIG['sentiment']['hypothesis_template'])
                sentiments.append(result['labels'][0])
                
                # Update progress for sentiment analysis
                current_progress = 80 + (idx / total_texts * 15)
                progress_bar.progress(int(current_progress))
                
            except Exception as e:
                sentiments.append("Neutral")
        
        # Split classifications
        status_text.text("Finalizing results...")
        progress_bar.progress(95)
        
        request_importance = []
        sentiment_values = []
        
        for classification in sentiments:
            if classification in ["highRating+", "highRating"]:
                request_importance.append(classification)
                sentiment_values.append("")
            else:
                request_importance.append("")
                sentiment_values.append(classification)
        
        # Add final columns
        filtered_domain_df['RequestFeatureImportance'] = request_importance
        filtered_domain_df['Sentiment'] = sentiment_values
        
        # Complete progress
        progress_bar.progress(100)
        status_text.text("Processing complete!")
        time.sleep(1)  # Show completion message briefly
        status_text.empty()  # Clear status message
        progress_bar.empty()  # Clear progress bar
        
        return filtered_domain_df
        
    except Exception as e:
        status_text.empty()
        progress_bar.empty()
        raise e

class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for the agent"""
    def on_llm_error(self, error: str, **kwargs) -> None:
        print(f"LLM Error: {error}")
    
    def on_tool_error(self, error: str, **kwargs) -> None:
        print(f"Tool Error: {error}")
    
    def on_agent_action(self, action, **kwargs) -> None:
        print(f"Agent Action: {action}")
    
    def on_chain_error(self, error: str, **kwargs) -> None:
        print(f"Chain Error: {error}")

def run_fuzzy_search_query(query, df, llm):
    """Run a fuzzy search query against the dataframe"""
    try:
        # Create a simplified dataframe for the agent with row numbers
        df_simple = df.copy()
        df_simple.reset_index(inplace=True)
        df_simple['row_number'] = df_simple.index
        df_simple['Created Date'] = pd.to_datetime(df_simple['Created Date']).dt.tz_localize('UTC')
        
        # Initialize the agent with clear instructions
        prefix = """You are working with a pandas dataframe to find matching rows.
You MUST follow this exact format in your responses:

Thought: I need to find rows that match the criteria
Action: python_repl_ast
Action Input: <pandas code that returns row numbers>
Observation: <result>
Thought: I now have the row numbers
Final Answer: <result>

Example patterns:
1. For domain-specific duplicates:
   df[(df['Solution Domain'].str.contains('data center networking', case=False, na=False)) & 
      ((df['possibleDuplicates'].str.len() > 0) | (df['CrossDomainDuplicates'].str.len() > 0))]['row_number'].tolist()

2. For sentiment in specific domain:
   df[(df['Solution Domain'].str.contains('domain name', case=False, na=False)) & 
      (df['Sentiment'] == 'Negative')]['row_number'].tolist()

3. For dates:
   df[(df['Created Date'] >= 'YYYY-MM-DD') & 
      (df['Created Date'] <= 'YYYY-MM-DD')]['row_number'].tolist()

4. For feature importance ratings:
   # For exact match of highRating+:
   df[df['RequestFeatureImportance'] == 'highRating+']['row_number'].tolist()
   # For exact match of highRating:
   df[df['RequestFeatureImportance'] == 'highRating']['row_number'].tolist()
   # For any highRating (both highRating and highRating+):
   df[df['RequestFeatureImportance'].isin(['highRating', 'highRating+'])]['row_number'].tolist()

5. For cross-domain duplicates with sentiment:
   # First get rows with cross-domain duplicates and negative sentiment
   cross_domain_rows = df[
       (df['CrossDomainDuplicates'].str.len() > 0) & 
       (df['Sentiment'] == 'Negative')
   ]
   # Extract original row numbers from CrossDomainDuplicates text
   original_rows = []
   duplicate_rows = cross_domain_rows['row_number'].tolist()
   for _, row in cross_domain_rows.iterrows():
       if 'Duplicate of Row' in row['CrossDomainDuplicates']:
           match = re.search(r'Row (\d+)', row['CrossDomainDuplicates'])
           if match:
               original_rows.append(int(match.group(1)))
   # Return both lists
   [duplicate_rows, original_rows]

Remember:
1. Always use python_repl_ast as the action
2. Put the pandas code in Action Input
3. NEVER modify the list of row numbers after getting them
4. Return the row numbers exactly as received from tolist()
5. For dates, use the format 'YYYY-MM-DD'
6. Use str.contains() with case=False, na=False for partial matches
7. Use exact matches (==) only when specified
8. For duplicates, check both possibleDuplicates and CrossDomainDuplicates columns
9. When searching domains, use the 'Solution Domain' column
10. For feature importance, use the 'RequestFeatureImportance' column and:
    - Use == for exact matches (e.g., exactly 'highRating+')
    - Use isin() for multiple values (e.g., both 'highRating' and 'highRating+')
11. For sentiment, use the 'Sentiment' column
12. For cross-domain duplicates:
    - First list should be duplicate row numbers
    - Second list should be original row numbers extracted from CrossDomainDuplicates text"""

        # Configure the agent with minimal prompt variables
        agent = create_pandas_dataframe_agent(
            llm,
            df_simple,
            prefix=prefix,
            max_iterations=CONFIG['agent']['max_iterations'],
            max_execution_time=CONFIG['agent']['max_execution_time'],
            verbose=True,
            handle_parsing_errors=True,
            include_df_in_prompt=False,
            allow_dangerous_code=True,
            number_of_head_rows=3
        )

        # Execute the query with minimal prompt variables
        response = agent.invoke({
            "input": f"Find rows matching this query: {query}",
            "df": df_simple  # Pass dataframe directly
        })
        
        # Extract the row numbers from the response
        output = response.get('output', '')
        print(f"Agent output: {output}")  # Debug print
        
        # Special handling for cross-domain duplicate queries
        if "cross domain duplicates" in query.lower():
            try:
                # Look for a list in the output
                start_idx = output.find('[')
                end_idx = output.rfind(']')
                if start_idx != -1 and end_idx != -1:
                    result_str = output[start_idx:end_idx+1]
                    result = eval(result_str)
                    
                    if isinstance(result, list) and len(result) == 2:
                        dup_numbers = result[0]
                        orig_numbers = result[1]
                        
                        # Get rows with explicit index handling
                        duplicate_rows = df_simple[df_simple['row_number'].isin(dup_numbers)].copy()
                        original_rows = df_simple[df_simple['Original_Row_Number'].isin(orig_numbers)].copy()
                        
                        # Combine results
                        result_df = pd.concat([duplicate_rows, original_rows], ignore_index=True)
                        result_df = result_df.drop_duplicates(subset=['Original_Row_Number', 'Reason_W_AddDetails'])
                        
                        if len(result_df) > 0:
                            result_df = result_df.drop(['row_number', 'index'], axis=1, errors='ignore')
                            return result_df.to_markdown(index=False)
                return "No matching results found."
            except Exception as e:
                print(f"Error processing cross-domain results: {str(e)}")
                return "Error processing cross-domain results."
        
        # Regular query processing
        matches = re.findall(r'\[[\d\s,]+\]', output)
        if matches:
            try:
                row_numbers = eval(matches[-1])
                if isinstance(row_numbers, list) and row_numbers:
                    result_df = df.iloc[row_numbers]
                    if len(result_df) > 0:
                        return result_df.to_markdown(index=False)
                return "No matching results found."
            except Exception as e:
                print(f"Error evaluating row numbers: {str(e)}")
                return f"Error: Failed to process row numbers: {str(e)}"
        
        return "No matching results found."
        
    except Exception as e:
        print(f"Error in query processing: {str(e)}")
        return f"Error: {str(e)}"

def run_fuzzy_search_query_with_retry(query, df, llm, max_retries=None):
    """Run fuzzy search query with retry mechanism"""
    if max_retries is None:
        max_retries = CONFIG['retry']['max_retries']
        
    progress_bar = st.progress(0, "Processing query...")
    status_text = st.empty()
    
    for attempt in range(max_retries):
        try:
            # Update progress
            progress = (attempt + 1) / max_retries
            progress_bar.progress(progress, f"Attempt {attempt + 1} of {max_retries}")
            status_text.text(f"Processing attempt {attempt + 1}...")
            
            # Run the query
            result = run_fuzzy_search_query(query, df, llm)
            
            if result:
                if result == "No matching results found.":
                    if attempt < max_retries - 1:
                        status_text.text(f"No results found, retrying... ({attempt + 2} of {max_retries})")
                        time.sleep(CONFIG['retry']['sleep_seconds'])
                        continue
                    progress_bar.empty()
                    status_text.empty()
                    return result
                elif result.startswith("Error:"):
                    if attempt < max_retries - 1:
                        status_text.text(f"Error occurred, retrying... ({attempt + 2} of {max_retries})")
                        time.sleep(CONFIG['retry']['sleep_seconds'])
                        continue
                    progress_bar.empty()
                    status_text.empty()
                    return result
                else:
                    # Success - we got a markdown table
                    progress_bar.empty()
                    status_text.empty()
                    return result
            
            # If we get here with no valid result and it's not the last attempt, retry
            if attempt < max_retries - 1:
                status_text.text(f"No valid result, retrying... ({attempt + 2} of {max_retries})")
                time.sleep(CONFIG['retry']['sleep_seconds'])
                continue
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                status_text.text(f"Error occurred, retrying... ({attempt + 2} of {max_retries})")
                time.sleep(CONFIG['retry']['sleep_seconds'])
            else:
                progress_bar.empty()
                status_text.empty()
                st.error(f"All retry attempts failed: {str(e)}")
                return "No matching results found."
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # If we get here, all retries failed
    return "No matching results found."

def clean_text_for_theme_analysis(text):
    """Clean and preprocess text for theme analysis"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', str(text))
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Replace multiple periods with single period
    text = re.sub(r'\.+', '.', text)
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.,!?])(\w)', r'\1 \2', text)
    
    return text.strip()

def extract_domain_texts(df, domain):
    """Extract and preprocess texts for a specific domain"""
    # Filter for domain
    domain_df = df[df['Solution Domain'] == domain].copy()
    
    # Extract and clean texts
    texts = []
    for _, row in domain_df.iterrows():
        cleaned_text = clean_text_for_theme_analysis(row['Reason_W_AddDetails'])
        if cleaned_text:  # Only add non-empty texts
            texts.append(cleaned_text)
    
    return texts

def format_text_preview(texts, max_preview=5):
    """Format texts for preview display"""
    total_texts = len(texts)
    preview_texts = texts[:max_preview]
    
    preview = "### Text Preview (first {} of {} entries)\n\n".format(
        min(max_preview, total_texts), 
        total_texts
    )
    
    for i, text in enumerate(preview_texts, 1):
        preview += f"{i}. {text}\n\n"
    
    if total_texts > max_preview:
        preview += f"... and {total_texts - max_preview} more entries"
    
    return preview

def analyze_themes_with_llm(texts, domain, top_n, llm):
    """Analyze texts using LLM to extract themes"""
    prompt = f"""Analyze the following text entries from the {domain} solution domain and identify the top {top_n} complaint themes.

For each theme, provide:
1. Clear title and occurrence count
2. Detailed description paragraph (2-3 sentences)
3. 3-4 specific examples from the provided text entries
4. Supporting evidence and impact analysis
5. Actionable recommendations

Your response MUST follow this exact format and MUST be complete:

# **Analysis of Top Complaint Themes in Solutions Domain {domain}**

[Generate {top_n} theme sections using this format for each]
## **[Theme Number]. [Theme Title]**  
**Occurrences**: [Number]

[Theme description paragraph]

### **Key Examples from the Dataset:**  
- **[Customer Name]** – [Example details] (Created: [Date], Account: [Account Name])
- **[Customer Name]** – [Example details] (Created: [Date], Account: [Account Name])
- **[Customer Name]** – [Example details] (Created: [Date], Account: [Account Name])

### **Supporting Evidence:**
- [Evidence point 1]
- [Evidence point 2]
- [Evidence point 3]

### **Impact:**
• **Business**: [Describe business impact]
• **Customer**: [Describe customer impact]
• **Operations**: [Describe operational impact]

### **Recommendations:**
1. [Short-term action item with timeline]
2. [Medium-term action item with priority]
3. [Long-term action item with expected outcome]

[After all themes, include this conclusion section]
# **Conclusion and Recommendations**

This analysis provides **valuable insights** into user pain points in {domain}. Based on the trends identified, the following **actionable recommendations** are proposed:

1. **Immediate Actions (0-3 months)**
   - [Action 1]
   - [Action 2]

2. **Medium-term Initiatives (3-6 months)**
   - [Initiative 1]
   - [Initiative 2]

3. **Long-term Strategic Goals (6+ months)**
   - [Goal 1]
   - [Goal 2]

Text entries to analyze:
{texts[:min(len(texts), 20)]}  # Limit number of examples to prevent token overflow
"""

    try:
        # Call LLM with retry mechanism
        for attempt in range(CONFIG['retry']['max_retries']):
            try:
                response = llm.invoke(prompt)
                if response and isinstance(response.content, str):
                    # Verify the response is complete
                    if "# **Conclusion and Recommendations**" not in response.content:
                        if attempt == CONFIG['retry']['max_retries'] - 1:
                            raise Exception("Generated report appears incomplete. Please try with fewer themes or a smaller date range.")
                        continue
                    return response.content
            except Exception as e:
                if attempt == CONFIG['retry']['max_retries'] - 1:
                    raise e
                time.sleep(CONFIG['retry']['sleep_seconds'])
        
        return None
    except Exception as e:
        print(f"Error in theme analysis: {str(e)}")
        raise e

def format_report_for_display(report_text):
    """Format the report for Streamlit display"""
    if not report_text:
        return ""
    
    # Add any custom formatting here if needed
    return report_text

def save_report_as_markdown(report_text, domain):
    """Save the report as a markdown file"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"reports/theme_analysis_{domain.replace(' ', '_')}_{timestamp}.md"
        
        # Save the report
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return filename
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        raise e

def create_pdf_report(report_text, domain):
    """Convert markdown report to PDF using ReportLab with improved formatting"""
    try:
        # Create PDF buffer
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Styles
        styles = getSampleStyleSheet()
        
        # Custom styles for better formatting
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#34495e')
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading3',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.HexColor('#2980b9')
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=8
        ))
        
        styles.add(ParagraphStyle(
            name='CustomBullet',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,
            leftIndent=20,
            firstLineIndent=-20,
            spaceAfter=8
        ))

        # Process markdown and create elements
        elements = []
        current_list = []
        
        # Split into lines
        lines = report_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_list:
                    elements.extend(current_list)
                    current_list = []
                elements.append(Spacer(1, 12))
                continue

            # Headers
            if line.startswith('# '):
                if current_list:
                    elements.extend(current_list)
                    current_list = []
                text = line[2:].strip('*').strip()
                elements.append(Paragraph(text, styles['CustomHeading1']))
            elif line.startswith('## '):
                if current_list:
                    elements.extend(current_list)
                    current_list = []
                text = line[3:].strip('*').strip()
                elements.append(Paragraph(text, styles['CustomHeading2']))
            elif line.startswith('### '):
                if current_list:
                    elements.extend(current_list)
                    current_list = []
                text = line[4:].strip('*').strip()
                elements.append(Paragraph(text, styles['CustomHeading3']))
            # Bullet points
            elif line.startswith('- ') or line.startswith('* '):
                text = line[2:].strip()
                # Handle customer examples differently
                if '**' in text and ' – ' in text:
                    parts = text.split(' – ')
                    if len(parts) == 2:
                        customer = parts[0].strip('*').strip()
                        details = parts[1].strip()
                        # Use <b> tag for bold customer name
                        formatted_text = f"• <b>{customer}</b> – {details}"
                        current_list.append(Paragraph(formatted_text, styles['CustomBullet']))
                # Handle Impact section categories
                elif ': ' in text and any(text.startswith(category) for category in ['Business:', 'Customer:', 'Operations:']):
                    parts = text.split(': ', 1)
                    category = parts[0]
                    details = parts[1] if len(parts) > 1 else ''
                    formatted_text = f"• <b>{category}</b>: {details}"
                    current_list.append(Paragraph(formatted_text, styles['CustomBullet']))
                else:
                    current_list.append(Paragraph(f"• {text}", styles['CustomBullet']))
            # Regular paragraphs
            else:
                if current_list:
                    elements.extend(current_list)
                    current_list = []
                elements.append(Paragraph(line, styles['CustomBody']))

        # Add any remaining list items
        if current_list:
            elements.extend(current_list)

        # Build PDF
        doc.build(elements)
        
        # Get the value of the BytesIO buffer
        pdf = buffer.getvalue()
        buffer.close()
        
        return pdf
        
    except Exception as e:
        print(f"Error creating PDF with ReportLab: {str(e)}")
        raise e

def show_loading_animation():
    """Show a loading animation with custom HTML/CSS"""
    st.markdown("""
        <style>
        .loading-spinner {
            margin: 20px 0;
            text-align: center;
        }
        .loading-spinner::after {
            content: '⏳';
            animation: loading 1.5s infinite;
            display: inline-block;
        }
        @keyframes loading {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div class="loading-spinner"></div>
    """, unsafe_allow_html=True)

def show_report_stats(report_text):
    """Show statistics about the generated report"""
    if not report_text:
        return
    
    # Count sections
    theme_count = len(re.findall(r'##\s+\*\*\d+\.', report_text))
    example_count = len(re.findall(r'-\s+\*\*', report_text))
    recommendation_count = len(re.findall(r'###\s+\*\*\d+\.', report_text))
    
    st.markdown("### 📊 Report Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Themes Analyzed", theme_count)
    with col2:
        st.metric("Examples Included", example_count)
    with col3:
        st.metric("Recommendations", recommendation_count)

if uploaded_file:
    # First just save the file and show sheet selection
    file_path = os.path.join('data', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Load Excel file and show sheet selection
    xl = pd.ExcelFile(file_path)
    sheet_name = st.sidebar.selectbox(
        "Select Sheet",
        options=xl.sheet_names,
        format_func=lambda x: f"Sheet: {x}"
    )
    
    if sheet_name:
        # Load data and show domain selection
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Fill empty Account Name cells with values from above
        account_cols = [col for col in df.columns if 'Account Name' in col]
        if account_cols:
            account_col = account_cols[0]
            df[account_col] = df[account_col].ffill()
        
        # Convert Created Date to datetime with robust error handling
        try:
            df['Created Date'] = pd.to_datetime(df['Created Date'], format='%m/%d/%Y', errors='coerce')
            if df['Created Date'].isna().any():
                df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
            
            valid_dates_mask = ~df['Created Date'].isna()
            df = df[valid_dates_mask].copy()
            
            if len(df) == 0:
                st.error("No valid data found after filtering dates")
                st.stop()
            
            min_date = df['Created Date'].min().date()
            max_date = df['Created Date'].max().date()
            
        except Exception as e:
            st.error(f"Error processing dates in 'Created Date' column. Expected format is M/D/YYYY (example: 2/16/2024). Error: {str(e)}")
            st.stop()
        
        unique_domains = df['Solution Domain'].unique().tolist()
        domain_options = ["ALL Domains"] + unique_domains
        
        selected_domain = st.sidebar.selectbox(
            "Select Solution Domain",
            options=domain_options,
            format_func=lambda x: f"Domain: {x}"
        )
        
        # Show time period selection for all domains
        time_period = st.sidebar.radio(
            "Select Time Period",
            ["Entire Time Period", "Custom Time Period"],
            key="time_period"
        )
        
        if time_period == "Custom Time Period":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                start_date = st.date_input("Start Date", 
                                         min_value=min_date,
                                         max_value=max_date,
                                         value=min_date,
                                         key="start_date")
            with col2:
                end_date = st.date_input("End Date",
                                       min_value=min_date,
                                       max_value=max_date,
                                       value=max_date,
                                       key="end_date")
            
            if start_date > end_date:
                st.sidebar.error("End date must be after start date")
                st.stop()
        
        if selected_domain and st.sidebar.button("Process Domain"):
            # First load models
            with st.spinner('Loading models...'):
                embedding_model, classifier = load_models()
            
            with st.spinner(f'Processing data for domain{"s" if selected_domain == "ALL Domains" else ""}: {selected_domain}...'):
                # Create progress tracking immediately
                main_progress = st.progress(0)
                main_status = st.empty()
                main_status.text("Starting processing...")
                
                try:
                    # Apply date filter if custom time period is selected
                    if time_period == "Custom Time Period":
                        main_status.text("Applying date filters...")
                        main_progress.progress(10)
                        start_datetime = pd.Timestamp(start_date)
                        end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                        df = df[(df['Created Date'] >= start_datetime) & (df['Created Date'] <= end_datetime)]
                        
                        if len(df) == 0:
                            st.error("No data found in the selected date range")
                            main_progress.empty()
                            main_status.empty()
                        
                        st.info(f"Filtered to {len(df)} records between {start_date} and {end_date}")
                    
                    if selected_domain == "ALL Domains":
                        main_status.text("Processing all domains...")
                        main_progress.progress(20)
                        # Process all domains
                        all_domains_df = pd.DataFrame()
                        all_texts = []
                        text_to_row_map = {}
                        
                        # Create progress tracking for multiple domains
                        domains_progress = st.progress(0)
                        domains_status = st.empty()
                        
                        # Track processed domains count for progress
                        processed_count = 0
                        total_domains = len(unique_domains)
                        
                        # First pass: Process each domain without duplicate detection
                        for domain in unique_domains:
                            domain_data = df[df['Solution Domain'] == domain]
                            if len(domain_data) == 0:
                                st.warning(f"Skipping empty domain: {domain}")
                                continue
                            
                            try:
                                # Update domain progress
                                progress_percent = (processed_count / total_domains) * 50
                                domains_progress.progress(int(progress_percent))
                                domains_status.text(f"Processing domain {processed_count + 1} of {total_domains}: {domain}")
                                main_status.text(f"Processing domain: {domain}")
                                main_progress.progress(int(20 + (progress_percent * 0.6)))
                                
                                # Process domain
                                domain_df = process_domain_data(df, domain, embedding_model, classifier, skip_duplicates=True)
                                
                                # Add domain identifier column at the start
                                domain_df.insert(0, 'Solution_Domain', domain)
                                start_idx = len(all_texts)
                                domain_texts = domain_df['Reason_W_AddDetails'].tolist()
                                all_texts.extend(domain_texts)
                                
                                for i, text in enumerate(domain_texts):
                                    text_to_row_map[start_idx + i] = {
                                        'domain': domain,
                                        'original_row': domain_df['Original_Row_Number'].iloc[i],
                                        'df_index': len(all_domains_df) + i
                                    }
                                
                                all_domains_df = pd.concat([all_domains_df, domain_df], ignore_index=True)
                                processed_count += 1
                                
                            except Exception as e:
                                st.error(f"Error processing domain {domain}: {str(e)}")
                                st.stop()
                        
                        # Second pass: Cross-domain duplicate detection
                        if len(all_texts) > 0:
                            domains_status.text("Performing cross-domain duplicate detection...")
                            
                            # Generate embeddings for all texts
                            normalized_embeddings = get_embeddings_batch(all_texts, embedding_model)
                            
                            # Compute similarity matrix
                            similarity_matrix = cosine_similarity(normalized_embeddings)
                            
                            # Initialize duplicates columns
                            all_domains_df['possibleDuplicates'] = ""
                            all_domains_df['CrossDomainDuplicates'] = ""
                            
                            # Find duplicates across all domains
                            threshold = CONFIG['duplicates']['similarity_threshold']
                            for i in range(len(all_texts)):
                                for j in range(i):
                                    if similarity_matrix[i, j] > threshold:
                                        duplicate_info = text_to_row_map[j]
                                        current_info = text_to_row_map[i]
                                        current_idx = current_info['df_index']
                                        original_idx = duplicate_info['df_index']
                                        
                                        # Create duplicate message
                                        duplicate_msg = f"Duplicate of Row {duplicate_info['original_row']} in domain '{duplicate_info['domain']}'"
                                        
                                        # Check if this is a cross-domain duplicate
                                        if duplicate_info['domain'] != current_info['domain']:
                                            all_domains_df.at[current_idx, 'CrossDomainDuplicates'] = duplicate_msg
                                            all_domains_df.at[original_idx, 'CrossDomainDuplicates'] = "Has duplicates in other domains"
                                        else:
                                            all_domains_df.at[current_idx, 'possibleDuplicates'] = duplicate_msg
                                            all_domains_df.at[original_idx, 'possibleDuplicates'] = "Has duplicates in this domain"
                                        break
                        
                        # Store results in session state
                        st.session_state.processed_df = all_domains_df
                        st.session_state.selected_domain = "ALL Domains"
                        st.success(f"Successfully processed {processed_count} out of {total_domains} domains")
                        
                    else:
                        # Process single domain
                        main_status.text(f"Processing domain: {selected_domain}")
                        main_progress.progress(30)
                        st.session_state.processed_df = process_domain_data(
                            df, selected_domain, embedding_model, classifier
                        )
                        st.session_state.selected_domain = selected_domain
                    
                    # Update progress for completion
                    main_status.text("Processing complete!")
                    main_progress.progress(100)
                    time.sleep(0.5)
                    main_progress.empty()
                    main_status.empty()
                    
                    # Clear other progress indicators
                    if 'domains_progress' in locals():
                        domains_progress.empty()
                    if 'domains_status' in locals():
                        domains_status.empty()
                    
                except Exception as e:
                    if 'main_progress' in locals():
                        main_progress.empty()
                    if 'main_status' in locals():
                        main_status.empty()
                    if 'domains_progress' in locals():
                        domains_progress.empty()
                    if 'domains_status' in locals():
                        domains_status.empty()
                    st.error(f"Error during processing: {str(e)}")
                    st.stop()

# Display results
if st.session_state.processed_df is not None:
    # Create tabs for different functionalities with state management
    tab_names = ["📊 Main Analysis", "🔍 Fuzzy Search", "📝 Reports"]
    main_tab, fuzzy_search_tab, reports_tab = st.tabs(tab_names)
    
    with main_tab:
        if st.session_state.selected_domain == "ALL Domains":
            st.markdown("### Analysis for ALL Solution Domains")
        else:
            st.markdown(f"### Analysis for Solutions Domain: {st.session_state.selected_domain}")
        st.markdown("---")
        
        # Add customer search functionality
        st.markdown("### Search by Customer")
        customer_search = st.text_input("Enter customer name to filter (case insensitive):")
        
        # Find the account name column
        account_cols = [col for col in st.session_state.processed_df.columns if 'Account Name' in col]
        if not account_cols:
            st.error("No column containing 'Account Name' found in the data")
        else:
            account_col = account_cols[0]  # Use the first column that contains 'Account Name'
            
            # Filter DataFrame based on customer search
            if customer_search:
                filtered_df = st.session_state.processed_df[
                    st.session_state.processed_df[account_col].str.contains(customer_search, case=False, na=False)
                ]
                if len(filtered_df) > 0:
                    st.success(f"Found {len(filtered_df)} entries matching '{customer_search}'")
                    # Display filtered DataFrame with custom styling
                    styled_filtered = filtered_df.style.format(na_rep='')
                    styled_filtered = styled_filtered.map(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
                    st.dataframe(styled_filtered, use_container_width=True)
                    
                    # Add download buttons for filtered results
                    col1, col2 = st.columns(2)
                    with col1:
                        # Excel download for filtered results
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            filtered_df.to_excel(writer, index=False)
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 Download Filtered Results as Excel",
                            data=excel_data,
                            file_name=f"filtered_{customer_search}_{st.session_state.selected_domain.lower().replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="filtered_excel"
                        )
                    
                    with col2:
                        # CSV download for filtered results
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Results as CSV",
                            data=csv,
                            file_name=f"filtered_{customer_search}_{st.session_state.selected_domain.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key="filtered_csv"
                        )
                else:
                    st.warning(f"No entries found matching '{customer_search}'")
        
        st.markdown("### Complete Analysis Results")
        # Display the complete processed DataFrame with custom styling
        styled_df = st.session_state.processed_df.style.format(na_rep='')
        styled_df = styled_df.map(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
        st.dataframe(styled_df, use_container_width=True)
        
        # Create columns for download buttons
        col1, col2 = st.columns(2)
        
        # Excel download
        with col1:
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                st.session_state.processed_df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            
            st.download_button(
                label="📥 Download as Excel",
                data=excel_data,
                file_name=f"processed_{st.session_state.selected_domain.lower().replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # CSV download
        with col2:
            csv = st.session_state.processed_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"processed_{st.session_state.selected_domain.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # Create and display duplicates DataFrame
        if st.session_state.selected_domain == "ALL Domains":
            duplicates_mask = (
                (st.session_state.processed_df['possibleDuplicates'].str.len() > 0) |
                (st.session_state.processed_df['CrossDomainDuplicates'].str.len() > 0)
            )
        else:
            duplicates_mask = (st.session_state.processed_df['possibleDuplicates'].str.len() > 0)
        
        duplicates_df = st.session_state.processed_df[duplicates_mask].copy()
        
        if len(duplicates_df) > 0:
            st.markdown("---")
            st.markdown("### Duplicate Entries Analysis")
            
            # Add customer search for duplicates
            st.markdown("### Search Duplicates by Customer")
            dup_customer_search = st.text_input("Enter customer name to filter duplicates (case insensitive):", key="dup_search")
            
            # Filter duplicates DataFrame based on customer search
            if dup_customer_search:
                filtered_dups = duplicates_df[
                    duplicates_df[account_col].str.contains(dup_customer_search, case=False, na=False)
                ]
                if len(filtered_dups) > 0:
                    st.success(f"Found {len(filtered_dups)} duplicate entries matching '{dup_customer_search}'")
                    # Display filtered duplicates with custom styling
                    styled_filtered_dups = filtered_dups.style.format(na_rep='')
                    styled_filtered_dups = styled_filtered_dups.map(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
                    st.dataframe(styled_filtered_dups, use_container_width=True)
                    
                    # Add download buttons for filtered duplicates
                    dup_col1, dup_col2 = st.columns(2)
                    with dup_col1:
                        # Excel download for filtered duplicates
                        dup_output = io.BytesIO()
                        with pd.ExcelWriter(dup_output, engine='openpyxl') as writer:
                            filtered_dups.to_excel(writer, index=False)
                        dup_excel_data = dup_output.getvalue()
                        
                        st.download_button(
                            label="📥 Download Filtered Duplicates as Excel",
                            data=dup_excel_data,
                            file_name=f"filtered_duplicates_{dup_customer_search}_{st.session_state.selected_domain.lower().replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key="filtered_dup_excel"
                        )
                    
                    with dup_col2:
                        # CSV download for filtered duplicates
                        dup_csv = filtered_dups.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Filtered Duplicates as CSV",
                            data=dup_csv,
                            file_name=f"filtered_duplicates_{dup_customer_search}_{st.session_state.selected_domain.lower().replace(' ', '_')}.csv",
                            mime="text/csv",
                            key="filtered_dup_csv"
                        )
                else:
                    st.warning(f"No duplicate entries found matching '{dup_customer_search}'")
            
            # Display complete duplicates analysis
            st.markdown("### Complete Duplicates Analysis")
            # Count different types of duplicates
            if st.session_state.selected_domain == "ALL Domains":
                original_entries = duplicates_df[
                    (duplicates_df['possibleDuplicates'].str.contains('Has duplicates', na=False)) |
                    (duplicates_df['CrossDomainDuplicates'].str.contains('Has duplicates', na=False))
                ]
                duplicate_entries = duplicates_df[
                    (duplicates_df['possibleDuplicates'].str.contains('Duplicate of', na=False)) |
                    (duplicates_df['CrossDomainDuplicates'].str.contains('Duplicate of', na=False))
                ]
            else:
                original_entries = duplicates_df[duplicates_df['possibleDuplicates'].str.contains('Has duplicates', na=False)]
                duplicate_entries = duplicates_df[duplicates_df['possibleDuplicates'].str.contains('Duplicate of', na=False)]
            
            st.markdown(f"""
            Found {len(duplicates_df)} entries in duplicate analysis:
            - {len(original_entries)} original entries that have duplicates
            - {len(duplicate_entries)} duplicate entries referencing other rows
            """)
            
            # Display complete duplicates DataFrame with custom styling
            styled_duplicates = duplicates_df.style.format(na_rep='')
            styled_duplicates = styled_duplicates.map(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
            st.dataframe(styled_duplicates, use_container_width=True)
            
            # Add download buttons for complete duplicates analysis
            dup_complete_col1, dup_complete_col2 = st.columns(2)
            
            # Excel download for complete duplicates
            with dup_complete_col1:
                dup_complete_output = io.BytesIO()
                with pd.ExcelWriter(dup_complete_output, engine='openpyxl') as writer:
                    duplicates_df.to_excel(writer, index=False)
                dup_complete_excel_data = dup_complete_output.getvalue()
                
                st.download_button(
                    label="📥 Download Complete Duplicates as Excel",
                    data=dup_complete_excel_data,
                    file_name=f"complete_duplicates_{st.session_state.selected_domain.lower().replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="complete_dup_excel"
                )
            
            # CSV download for complete duplicates
            with dup_complete_col2:
                dup_complete_csv = duplicates_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Duplicates as CSV",
                    data=dup_complete_csv,
                    file_name=f"complete_duplicates_{st.session_state.selected_domain.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="complete_dup_csv"
                )

    with fuzzy_search_tab:
        st.markdown("### Fuzzy Search Query")
        
        # Example queries in collapsible box
        with st.expander("📝 Example Queries"):
            st.markdown("""
            - Show me ALL rows that have Negative sentiment across ALL solution domains
            - Show me the records between March 15th 2024 and March 16th 2024
            - Show me all rows for possible duplicates for data center networking
            - Show me the rows where the account name matches 'BU - MAN DE'
            - Show me the rows where the account name has at&t and sentiment is negative
            - Show me all rows that are duplicates and also solutions domain has campus and have negative sentiment and created in April 2024
            - Show me all rows that have highrating
            - Show me all rows that have highrating+
            - show me all the rows that have exactly highrating
            - show me all the rows that have exactly highrating and also duplicates 
            - Show me the records that have cross domain duplicates and sentiment has negative, then from the cross domain duplicates get the row number that is matched and show me that row number for the Original Row Number column along with the cross domain duplicate rows
            """)
        
        # Create the fuzzy search interface
        query = st.text_area(
            "Enter your search query:",
            height=100,
            placeholder="e.g., Show me all rows that have Negative sentiment",
            key="fuzzy_search_query"  # Add unique key
        )
        
        if st.button("🔍 Search", type="primary", key="fuzzy_search_button"):  # Add unique key
            if query:
                try:
                    # Set active tab to fuzzy search
                    st.session_state.active_tab = 1
                    
                    # Initialize a fresh LLM instance for each query
                    with st.spinner("Initializing AI model..."):
                        try:
                            llm = init_azure_openai()
                        except Exception as e:
                            st.error(f"Error initializing AI model: {str(e)}")
                            st.stop()
                    
                    # Run the fuzzy search query with retry mechanism
                    result = run_fuzzy_search_query_with_retry(
                        query, 
                        st.session_state.processed_df, 
                        llm
                    )
                    
                    # Only proceed with parsing and display if we got a result
                    if result is not None:
                        if result == "No matching results found.":
                            st.warning("No results found for your query")
                        else:
                            df_result, error = parse_markdown_table(result)
                            if df_result is not None:
                                display_fuzzy_search_results(df_result)
                            else:
                                st.warning("No results found for your query")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
            else:
                st.warning("Please enter a search query")

    with reports_tab:
        st.markdown("### Theme Analysis Report Generator")
        
        # Add help text in expander
        with st.expander("ℹ️ How to use the Theme Analysis", expanded=False):
            st.markdown("""
            1. **Select Domain**: Choose the solution domain to analyze
            2. **Set Theme Count**: Choose how many top themes to identify (2-10)
            3. **Generate Report**: Click to analyze the data and create a themed report
            4. **Download**: Get the report in Markdown or PDF format
            
            The analysis will identify key themes, provide examples, and suggest recommendations.
            """)
        
        st.markdown("---")
        
        # Initialize session state for reports tab
        if 'report_domain' not in st.session_state:
            st.session_state.report_domain = None
        if 'top_n_themes' not in st.session_state:
            st.session_state.top_n_themes = 4
        if 'current_report' not in st.session_state:
            st.session_state.current_report = None
        if 'processing_error' not in st.session_state:
            st.session_state.processing_error = None
        
        # Domain selection in a container
        with st.container():
            st.subheader("1. Select Analysis Parameters")
            
            # Domain selection
            unique_domains = st.session_state.processed_df['Solution Domain'].unique().tolist()
            selected_report_domain = st.selectbox(
                "Select Solution Domain for Analysis",
                options=unique_domains,
                key="report_domain_select"
            )
            
            # Number of themes selection
            top_n = st.number_input(
                "Number of Top Themes to Analyze",
                min_value=2,
                max_value=10,
                value=st.session_state.top_n_themes,
                step=1,
                help="Select how many top themes to analyze (between 2 and 10)"
            )
        
        # Show entry count for selected domain
        if selected_report_domain:
            domain_entries = len(st.session_state.processed_df[
                st.session_state.processed_df['Solution Domain'] == selected_report_domain
            ])
            if domain_entries > 0:
                st.success(f"📊 Found {domain_entries} entries for analysis in {selected_report_domain}")
            else:
                st.warning("⚠️ No entries found for selected domain")
        
        st.markdown("---")
        st.subheader("2. Generate Analysis")

        # Generate Report button
        if st.button("🔄 Generate Theme Analysis Report", type="primary", use_container_width=True):
            if not selected_report_domain:
                st.error("❌ Please select a domain first")
            elif domain_entries == 0:
                st.error("❌ No entries found for selected domain")
            else:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                try:
                    # Step 1: Process Text
                    with progress_placeholder.container():
                        st.progress(0, "Starting analysis...")
                    status_placeholder.info("🔄 Processing domain text...")
                    
                    processed_texts = extract_domain_texts(
                        st.session_state.processed_df,
                        selected_report_domain
                    )
                    
                    if not processed_texts:
                        st.error("❌ No valid text entries found for analysis")
                        st.stop()
                    
                    # Update progress
                    with progress_placeholder.container():
                        st.progress(33, "Text processing complete")
                    
                    # Step 2: Initialize LLM
                    status_placeholder.info("🤖 Initializing AI model...")
                    llm = init_azure_openai()
                    
                    # Update progress
                    with progress_placeholder.container():
                        st.progress(66, "AI model ready")
                    
                    # Step 3: Generate Report
                    status_placeholder.info("📝 Generating theme analysis...")
                    text_content = "\n\n".join(processed_texts)
                    report = analyze_themes_with_llm(
                        text_content,
                        selected_report_domain,
                        top_n,
                        llm
                    )
                    
                    if report:
                        # Update progress to complete
                        with progress_placeholder.container():
                            st.progress(100, "Analysis complete!")
                        status_placeholder.empty()
                        
                        # Store report in session state
                        st.session_state.current_report = report
                        st.session_state.report_domain = selected_report_domain
                        st.session_state.processing_error = None
                        
                        # Save report to file
                        report_file = save_report_as_markdown(report, selected_report_domain)
                        
                        # Display success message
                        st.success("✅ Theme analysis complete!")
                    else:
                        st.error("❌ Failed to generate report. Please try again.")
                
                except Exception as e:
                    st.session_state.processing_error = str(e)
                    st.error(f"❌ Error during analysis: {str(e)}")
                
                finally:
                    # Clean up progress indicators
                    progress_placeholder.empty()
                    status_placeholder.empty()
        
        # Show error message if exists
        if st.session_state.processing_error:
            st.error(f"❌ Last Error: {st.session_state.processing_error}")
        
        # Display the current report if it exists and matches the selected domain
        if st.session_state.current_report and st.session_state.report_domain == selected_report_domain:
            # Show report statistics
            st.markdown("### 📊 Report Overview")
            show_report_stats(st.session_state.current_report)
            
            # Create download section
            st.markdown("### 📥 Download Report")
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                # Markdown download button
                st.download_button(
                    label="📝 Download as Markdown",
                    data=st.session_state.current_report,
                    file_name=f"theme_analysis_{selected_report_domain.replace(' ', '_')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="download_markdown_1"
                )
            
            with download_col2:
                # PDF download button using ReportLab
                try:
                    pdf_content = create_pdf_report(st.session_state.current_report, selected_report_domain)
                    st.download_button(
                        label="📄 Download as PDF",
                        data=pdf_content,
                        file_name=f"theme_analysis_{selected_report_domain.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="download_pdf_1"
                    )
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
            
            # Display the full report
            st.markdown("### 📑 Full Report")
            st.markdown(format_report_for_display(st.session_state.current_report))

# Footer with copyright
st.markdown("---")
st.markdown("© 2024 Solutions Domain Analyzer") 
