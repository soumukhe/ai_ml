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

# Force reload of environment variables
load_dotenv(override=True)

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
        temperature=0,
        max_tokens=1000,
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

        # First check for Final Answer
        if 'Final Answer:' in markdown_text:
            # Get everything after Final Answer
            table_text = markdown_text.split('Final Answer:')[1].strip()
        else:
            table_text = markdown_text
        
        # Split into lines and clean up
        lines = [line.strip() for line in table_text.split('\n') if line.strip() and '|' in line]
        
        if not lines:
            return None, None
            
        # Extract headers
        headers = [col.strip() for col in lines[0].split('|')[1:-1]]
        
        # Process data rows (skip header and separator)
        data = []
        for line in lines[2:]:  # Skip header and separator line
            if line.strip() and '|' in line:
                row = [col.strip() for col in line.split('|')[1:-1]]
                if len(row) == len(headers):  # Only add rows that match header length
                    data.append(row)
        
        # Create DataFrame
        if data:
            df = pd.DataFrame(data, columns=headers)
            return df, None
        return None, None
    except Exception as e:
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
    - **Auth**: BridgeIT (Cisco Internal)
    - ✅ Secure enterprise authentication
    - ✅ Internal infrastructure only
    - ✅ No public cloud usage
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
    # Simpler model loading without suppression
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model.to('cpu')  # Explicitly use CPU for consistency
    
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        clean_up_tokenization_spaces=True,
                        multi_label=True,
                        device='cpu')  # Explicitly use CPU
    return embedding_model, classifier

def get_embeddings_batch(texts, model, batch_size=32):
    """Get embeddings for a batch of texts using Hugging Face model"""
    all_embeddings = []
    total_expected = len(texts)
    
    # Clean all texts first
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Process in batches - simplified batch processing
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        if batch:  # Only process if batch is not empty
            try:
                # Get embeddings directly without filtering
                embeddings = model.encode(batch, 
                                       normalize_embeddings=True,
                                       show_progress_bar=False)
                all_embeddings.extend(embeddings)
            except Exception as e:
                st.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Fill with zeros for failed batch
                all_embeddings.extend([np.zeros(384) for _ in range(len(batch))])
    
    # Ensure we have exactly the right number of embeddings
    result = np.array(all_embeddings[:total_expected])
    return result

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
            threshold = 0.95
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
        sentiment_labels = ["highRating+", "highRating", "Neutral", "Negative", "Negative-"]
        sentiments = []
        
        # Process sentiments with incremental progress
        total_texts = len(filtered_domain_df['Reason_W_AddDetails'])
        for idx, text in enumerate(filtered_domain_df['Reason_W_AddDetails']):
            try:
                result = classifier(text, 
                                 candidate_labels=sentiment_labels,
                                 hypothesis_template="This feature request is {}.")
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

def run_fuzzy_search_query(query, df, llm):
    """Execute a fuzzy search query on the DataFrame"""
    try:
        with open('terminal_output.txt', 'a') as log_file:
            log_file.write(f"\nProcessing query: {query}\n")
            
            # Create simplified dataframe for the agent
            simplified_df = df.copy()
            simplified_df['row_number'] = range(len(df))
            simplified_df['Created Date'] = pd.to_datetime(simplified_df['Created Date'])
            
            finder_agent = create_pandas_dataframe_agent(
                llm=llm,  # Use the passed-in LLM instance
                df=simplified_df,
                verbose=True,
                max_iterations=3,
                max_execution_time=30.0,
                allow_dangerous_code=True,
                include_df_in_prompt=True,
                prefix="""You are working with a pandas dataframe that has these columns: Solution Domain, Account Name, Created Date, Product, Use Case, Created By, Status, Closed Date, Solution Domain, Next Step, Original_Row_Number, Reason_W_AddDetails, RequestFeatureImportance, Sentiment, possibleDuplicates, CrossDomainDuplicates.

Example queries and their patterns:
- "Show me all rows that have duplicates":
  df[((df['possibleDuplicates'].fillna('').str.len() > 0) | (df['CrossDomainDuplicates'].fillna('').str.len() > 0))]['row_number'].tolist()

- "Show me all rows that have exactly Negative sentiment":
  df[df['Sentiment'] == 'Negative']['row_number'].tolist()

- "Show me all rows where Solution Domain contains campus":
  df[df['Solution Domain'].str.lower().str.contains('campus', na=False)]['row_number'].tolist()

- "Show me all rows between March 15th and March 16th 2024":
  df[(df['Created Date'] >= '2024-03-15') & (df['Created Date'] <= '2024-03-16')]['row_number'].tolist()

- "Show me all rows where Account Name contains cisco":
  df[df['Account Name: Account Name  ↑'].str.lower().str.contains('cisco', case=False, na=False)]['row_number'].tolist()

- "Show me all rows that have exactly highRating":
  df[df['RequestFeatureImportance'] == 'highRating']['row_number'].tolist()

- "Show me all rows that have exactly highRating+":
  df[df['RequestFeatureImportance'] == 'highRating+']['row_number'].tolist()

- "Show me all rows that have exactly highRating+ and duplicates":
  df[(df['RequestFeatureImportance'] == 'highRating+') & ((df['possibleDuplicates'].fillna('').str.len() > 0) | (df['CrossDomainDuplicates'].fillna('').str.len() > 0))]['row_number'].tolist()

IMPORTANT: 
1. Return ONLY the list of row numbers
2. Do not append the results to the command
3. If the query contains the word 'exactly', use exact match (==) with EXACT case matching
4. If the query does not contain 'exactly', use case-insensitive contains
5. Always handle NA values with na=False in str operations
6. For RequestFeatureImportance and Sentiment columns, always use exact case matching (highRating, highRating+, Negative, etc.)""")
            
            log_file.write(f"Finding matching row numbers...\n")
            
            try:
                # Get matching row numbers
                response = finder_agent.invoke({
                    "input": f"Find matching rows for this query: {query}. Return ONLY the list of row numbers in the format [n1, n2, ...]. Do not include any other text or explanation."
                })
                
                # Extract and clean output
                output = response['output'] if isinstance(response, dict) else str(response)
                log_file.write(f"Raw output: {output}\n")
                
                # Look for list pattern [n1, n2, n3]
                import re
                list_matches = re.findall(r'\[([0-9, ]+)\]', output)
                if list_matches:
                    # Take the first list found
                    numbers_str = list_matches[0]
                    row_numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip()]
                    log_file.write(f"Extracted row numbers: {row_numbers}\n")
                    
                    if row_numbers:
                        # Get the matching rows and convert to markdown
                        result_df = df.iloc[row_numbers]
                        log_file.write(f"Found {len(row_numbers)} matching rows\n")
                        return result_df.to_markdown(index=False)
                
                # If we get here, no valid results were found
                log_file.write("No matching results found\n")
                return "No matching results found."
                
            except Exception as e:
                log_file.write(f"Error in agent execution: {str(e)}\n")
                return f"Error: {str(e)}"
                
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        with open('terminal_output.txt', 'a') as log_file:
            log_file.write(f"\n{error_msg}\n")
        return f"Error: {str(e)}"

def run_fuzzy_search_query_with_retry(query, df, llm, max_retries=3):
    """Run fuzzy search query with retry mechanism"""
    progress_bar = st.progress(0, "Processing query...")
    status_text = st.empty()
    
    for attempt in range(max_retries):
        try:
            # Update progress
            progress_bar.progress((attempt + 1) / max_retries, f"Attempt {attempt + 1} of {max_retries}")
            status_text.text(f"Processing attempt {attempt + 1}...")
            
            # Run the query with a timeout
            result = run_fuzzy_search_query(query, df, llm)
            
            # If we got a valid result (not None and not an error message)
            if result and isinstance(result, str):
                if result == "No matching results found.":
                    progress_bar.empty()
                    status_text.empty()
                    return result
                elif not result.startswith("Error:"):
                    progress_bar.empty()
                    status_text.empty()
                    return result  # Return immediately on success
                elif result.startswith("Error:"):
                    # Only retry on errors if not the last attempt
                    if attempt < max_retries - 1:
                        status_text.text(f"Error occurred, retrying... ({attempt + 2} of {max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        return result  # Return error on last attempt
            
            # If we get here with no valid result and it's not the last attempt, retry
            if attempt < max_retries - 1:
                status_text.text(f"No valid result, retrying... ({attempt + 2} of {max_retries})")
                time.sleep(2)
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                status_text.text(f"Error occurred, retrying... ({attempt + 2} of {max_retries})")
                time.sleep(2)
            else:
                progress_bar.empty()
                status_text.empty()
                raise e
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # If we get here, all retries failed
    return "No matching results found."

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
                            st.stop()
                        
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
                                continue
                        
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
                            threshold = 0.95
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
    tab_names = ["📊 Main Analysis", "🔍 Fuzzy Search"]
    main_tab, fuzzy_search_tab = st.tabs(tab_names)
    
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
            """)
        
        # Initialize LLM if not already done
        if st.session_state.llm is None:
            with st.spinner("Initializing AI model..."):
                try:
                    st.session_state.llm = init_azure_openai()
                except Exception as e:
                    st.error(f"Error initializing AI model: {str(e)}")
                    st.stop()
        
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
                    
                    # Run the fuzzy search query with retry mechanism
                    result = run_fuzzy_search_query_with_retry(
                        query, 
                        st.session_state.processed_df, 
                        st.session_state.llm
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

# Footer with copyright
st.markdown("---")
st.markdown("© 2024 Solutions Domain Analyzer") 