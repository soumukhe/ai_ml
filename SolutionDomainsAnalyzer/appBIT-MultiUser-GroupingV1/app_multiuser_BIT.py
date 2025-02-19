import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
import base64
import requests
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
import time
import re
import torch
from io import BytesIO
from transformers import pipeline
from tqdm import tqdm
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
import markdown
import traceback
import hashlib
import psutil
import matplotlib.pyplot as plt

# Import our custom modules
from user_manager import UserManager
from file_manager import FileManager
from login import login_page, get_current_user, init_session_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force reload of environment variables
load_dotenv(override=True)

# Get environment variables without defaults
app_key = os.getenv('app_key')
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
langsmith_api_key = os.getenv('LANGSMITH_API_KEY')

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
            return token_response.json()["access_token"]
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

def save_processed_data(df):
    """Save processed DataFrame to user directory in multiple formats"""
    current_user = get_current_user()
    if current_user:
        user_dir = st.session_state.user_manager.get_user_dir(current_user)
        if user_dir and df is not None:
            try:
                # Create backup of existing files if they exist
                processed_df_path = user_dir / "processed_data.pkl"
                csv_path = user_dir / "processed_data.csv"
                
                if processed_df_path.exists():
                    backup_pkl = user_dir / "processed_data.pkl.bak"
                    shutil.copy2(processed_df_path, backup_pkl)
                    
                if csv_path.exists():
                    backup_csv = user_dir / "processed_data.csv.bak"
                    shutil.copy2(csv_path, backup_csv)
                
                # Save new data
                df.to_pickle(processed_df_path)
                df.to_csv(csv_path, index=False)
                
                # Update session state
                st.session_state.processed_df = df
                st.session_state.processed_df_columns = list(df.columns)
                
                logger.info(f"Successfully saved processed data for user {current_user}")
                return True
            except Exception as e:
                logger.error(f"Error saving processed data: {str(e)}")
                return False
    return False

def initialize_user_managers():
    """Initialize file manager for current user"""
    logger.info("Starting user managers initialization")
    current_user = get_current_user()
    logger.info(f"Current user: {current_user}")
    
    if current_user:
        user_dir = st.session_state.user_manager.get_user_dir(current_user)
        logger.info(f"User directory: {user_dir}")
        
        if user_dir:
            st.session_state.file_manager = FileManager(user_dir)
            logger.info("File manager initialized")
            
            # Load previously processed data if available
            try:
                processed_df_path = user_dir / "processed_data.pkl"
                csv_path = user_dir / "processed_data.csv"
                
                logger.info(f"Looking for processed data at: {processed_df_path} or {csv_path}")
                
                if processed_df_path.exists():
                    logger.info("Found pickle file, attempting to load...")
                    try:
                        st.session_state.processed_df = pd.read_pickle(processed_df_path)
                        logger.info(f"Successfully loaded data from pickle with shape: {st.session_state.processed_df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading pickle file: {str(e)}")
                        st.session_state.processed_df = None
                
                # If pickle load failed or file doesn't exist, try CSV
                if st.session_state.processed_df is None and csv_path.exists():
                    logger.info("Attempting to load from CSV backup...")
                    try:
                        st.session_state.processed_df = pd.read_csv(csv_path)
                        logger.info(f"Successfully loaded data from CSV with shape: {st.session_state.processed_df.shape}")
                    except Exception as e:
                        logger.error(f"Error loading CSV file: {str(e)}")
                        st.session_state.processed_df = None
                
                if st.session_state.processed_df is not None:
                    st.session_state.selected_domain = "ALL Domains"
                    logger.info("Successfully loaded processed data")
                else:
                    logger.info("No processed data found from previous session")
                    st.session_state.selected_domain = None
                
            except Exception as e:
                logger.error(f"Error in data loading process: {str(e)}")
                st.session_state.processed_df = None
                st.session_state.selected_domain = None
            
            # Log final state
            logger.info(f"Initialization complete. DataFrame loaded: {st.session_state.processed_df is not None}")
            if st.session_state.processed_df is not None:
                logger.info(f"DataFrame shape: {st.session_state.processed_df.shape}")
                logger.info(f"DataFrame columns: {list(st.session_state.processed_df.columns)}")

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
        
        # Initialize duplicate columns with empty strings
        filtered_domain_df['possibleDuplicates'] = ""
        filtered_domain_df['CrossDomainDuplicates'] = ""
        
        if not skip_duplicates:
            # Generate embeddings for current domain
            status_text.text("Generating text embeddings for current domain...")
            progress_bar.progress(40)
            texts = filtered_domain_df['Reason_W_AddDetails'].tolist()
            normalized_embeddings = get_embeddings_batch(texts, embedding_model)
            
            # Store embeddings in session state
            st.session_state.domain_embeddings[domain] = {
                'texts': texts,
                'embeddings': normalized_embeddings
            }
            
            # Get all other domains' data for cross-domain comparison
            status_text.text("Preparing cross-domain comparison...")
            progress_bar.progress(50)
            other_domains_df = df[df['Solution Domain'] != domain].copy()
            
            if len(other_domains_df) > 0:
                # Generate embeddings for other domains
                status_text.text("Generating embeddings for other domains...")
                progress_bar.progress(60)
                other_texts = other_domains_df['Reason_W_AddDetails'].tolist()
                other_embeddings = get_embeddings_batch(other_texts, embedding_model)
                
                # Compute similarity matrices
                status_text.text("Computing similarity matrices...")
                progress_bar.progress(70)
                
                # Within-domain similarity
                similarity_matrix = cosine_similarity(normalized_embeddings)
                
                # Cross-domain similarity
                cross_similarity = cosine_similarity(normalized_embeddings, other_embeddings)
                
                # Find duplicates
                status_text.text("Detecting possible duplicates...")
                progress_bar.progress(80)
                threshold = CONFIG['duplicates']['similarity_threshold']
                
                # Create a mapping of filtered index to original row number
                row_number_map = dict(enumerate(filtered_domain_df['Original_Row_Number']))
                
                # Process both within-domain and cross-domain duplicates
                for i in range(len(texts)):
                    # Check within same domain first
                    within_domain_matches = []
                    for j in range(i):
                        if similarity_matrix[i, j] > threshold:
                            original_row = row_number_map[j]
                            within_domain_matches.append(f"Row {original_row}")
                    
                    if within_domain_matches:
                        filtered_domain_df.at[filtered_domain_df.index[i], 'possibleDuplicates'] = f"Duplicate of {', '.join(within_domain_matches)}"
                    
                    # Check cross-domain duplicates
                    cross_matches = []
                    cross_match_indices = np.where(cross_similarity[i] > threshold)[0]
                    for match_idx in cross_match_indices:
                        match_domain = other_domains_df.iloc[match_idx]['Solution Domain']
                        match_row = other_domains_df.iloc[match_idx]['Original_Row_Number']
                        cross_matches.append(f"Row {match_row} in domain '{match_domain}'")
                    
                    if cross_matches:
                        filtered_domain_df.at[filtered_domain_df.index[i], 'CrossDomainDuplicates'] = f"Duplicate of {', '.join(cross_matches)}"
        
        # Process sentiments
        status_text.text("Analyzing text sentiment and request importance...")
        progress_bar.progress(90)
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
                current_progress = 90 + (idx / total_texts * 8)
                progress_bar.progress(int(current_progress))
                
            except Exception as e:
                sentiments.append("Neutral")
        
        # Split classifications
        status_text.text("Finalizing results...")
        progress_bar.progress(98)
        
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

def analyze_solutions(df):
    """Analyze solutions domains and create visualizations"""
    try:
        # Group by Solution Domain and count occurrences
        domain_counts = df['Solution Domain'].value_counts()
        
        # Create bar chart
        fig_domains = px.bar(
            x=domain_counts.index,
            y=domain_counts.values,
            title='Solution Domain Distribution',
            labels={'x': 'Solution Domain', 'y': 'Count'}
        )
        
        # Customize layout
        fig_domains.update_layout(
            showlegend=False,
            xaxis_tickangle=45,
            height=500
        )
        
        return fig_domains
    except Exception as e:
        st.error(f"Error analyzing solutions: {str(e)}")
        return None

def generate_report(df, filename="solution_domain_report.pdf"):
    """Generate PDF report of analysis"""
    try:
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        story.append(Paragraph("Solution Domain Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        # Add summary statistics
        story.append(Paragraph("Summary Statistics", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Create summary table
        summary_data = [
            ['Total Records', str(len(df))],
            ['Unique Solution Domains', str(df['Solution Domain'].nunique())],
            ['Date Range', f"{df['Created Date'].min()} to {df['Created Date'].max()}"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Build PDF
        doc.build(story)
        return True
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return False

def clean_text(text):
    """Clean and validate text for API submission"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s.,!?-]', ' ', str(text))
    text = ' '.join(text.split())
    return text

def get_embeddings_batch(texts, model, batch_size=32, progress_callback=None):
    """Get embeddings for a batch of texts with proper normalization"""
    try:
        # Validate input texts
        if not texts:
            raise ValueError("No texts provided for embedding generation")
            
        # Filter out empty or invalid texts
        valid_texts = [str(text).strip() for text in texts if pd.notna(text) and str(text).strip()]
        if not valid_texts:
            raise ValueError("No valid texts found after filtering empty/NaN values")
            
        # Create progress indicators
        embedding_progress = st.progress(0, "Initializing embeddings...")
        embedding_status = st.empty()
        
        # Calculate total batches
        total_texts = len(valid_texts)
        total_batches = (total_texts + batch_size - 1) // batch_size
        
        # Process in batches with detailed progress updates
        all_embeddings = []
        for i in range(0, total_texts, batch_size):
            batch = valid_texts[i:i+batch_size]
            if not batch:
                continue
            
            # Update progress before processing batch
            current_batch = i // batch_size + 1
            progress = min(1.0, (i + batch_size) / total_texts)
            msg = f"Batch {current_batch}/{total_batches} ({min(i + batch_size, total_texts)}/{total_texts} texts)"
            
            embedding_progress.progress(progress, msg)
            embedding_status.text(f"Processing embeddings... {msg}")
            
            # Process batch with normalization enabled
            batch_embeddings = model.encode(
                batch,
                normalize_embeddings=True,  # Enable normalization during encoding
                show_progress_bar=False
            )
            
            if len(batch_embeddings) > 0:
                all_embeddings.extend(batch_embeddings)
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(progress, msg)
        
        if not all_embeddings:
            raise ValueError("No embeddings generated from the provided texts")
            
        # Stack all embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        # Clear progress indicators
        embedding_progress.empty()
        embedding_status.empty()
        
        return embeddings
        
    except Exception as e:
        # Clear progress indicators on error
        if 'embedding_progress' in locals():
            embedding_progress.empty()
        if 'embedding_status' in locals():
            embedding_status.empty()
        st.error(f"Error in embedding generation: {str(e)}")
        # Return zero vectors as fallback
        return np.zeros((len(texts), model.get_sentence_embedding_dimension()))

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
        
        # Load embedding model - use SentenceTransformer for both purposes
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

def parse_markdown_table(markdown_text):
    """Parse markdown table into a pandas DataFrame.
    
    Args:
        markdown_text (str): The markdown table text
        
    Returns:
        tuple: (pd.DataFrame, str) - The parsed DataFrame and any error message
    """
    try:
        # Check for no results message
        if not markdown_text or "No matching results found" in markdown_text:
            return None, "No matching results found"
            
        # Split into lines and clean up
        lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
        
        if not lines or len(lines) < 3:  # Need at least header, separator, and one data row
            return None, "Invalid table format: insufficient rows"
            
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
            
        return None, "No data rows found in table"
        
    except Exception as e:
        error_msg = f"Error parsing markdown table: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

def display_fuzzy_search_results(df_result):
    """Display fuzzy search results with download options"""
    if df_result is None or len(df_result) == 0:
        st.warning("No results found for your query")
        return
        
    st.success(f"Found {len(df_result)} matching records")
    # Display dataframe without index
    st.dataframe(df_result, use_container_width=True)
    
    # Add download buttons
    col1, col2 = st.columns(2)
    with col1:
        # Excel download without index
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
        # CSV download without index
        csv = df_result.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="search_results.csv",
            mime="text/csv"
        )

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
Observation: <r>
Thought: I now have the row numbers
Final Answer: <r>

Remember to STOP after getting the Final Answer. Do not continue processing.


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


        # Create Python REPL tool
        repl = PythonREPL()

        # Configure the agent
        tools = [
            Tool(
                name="python_repl_ast",
                description="Execute python code using ast.literal_eval for safety",
                func=repl.run
            )
        ]

        # Create the agent with strict limits
        agent = create_pandas_dataframe_agent(
            llm,
            df_simple,
            prefix=prefix,
            max_iterations=3,
            max_execution_time=30,
            verbose=True,
            tools=tools,
            include_df_in_prompt=True,
            handle_parsing_errors=True,
            input_variables=["df", "input", "agent_scratchpad"],
            top_k=10,
            allow_dangerous_code=True
        )

        # Execute the query with timeout
        response = agent.invoke({
            "input": f"Find rows matching this query: {query}",
            "df": df_simple
        })
        
        # Check for "Finished chain" immediately in the response
        if "Finished chain" in str(response):
            return "No matching results found."
            
        # Extract the output from the response
        output = response.get('output', '')
        
        # Look for row numbers in square brackets
        matches = re.findall(r'\[[\d\s,]+\]', output)
        if matches:
            try:
                row_numbers = eval(matches[-1])  # Get the last list of numbers
                if isinstance(row_numbers, list) and row_numbers:
                    # Get the rows from the original dataframe
                    result_df = df.iloc[row_numbers].copy()
                    if len(result_df) > 0:
                        return result_df.to_markdown(index=False)
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
            
            # Clear progress indicators before processing result
            progress_bar.empty()
            status_text.empty()
            
            # Check if we got a valid result
            if result and result != "No matching results found." and not result.startswith("Error:"):
                # Success - we got a markdown table
                return result
            
            # Check if the chain finished successfully
            if "Finished chain" in str(result):
                # Chain completed successfully, return the result even if no matches
                return result
            
            # Handle retries for unsuccessful results
            if attempt < max_retries - 1:
                if result == "No matching results found.":
                    time.sleep(CONFIG['retry']['sleep_seconds'])
                elif result.startswith("Error:"):
                    time.sleep(CONFIG['retry']['sleep_seconds'])
                continue
            
            # Return last result if we've exhausted retries
            return result
                
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(CONFIG['retry']['sleep_seconds'])
            else:
                # Clear progress indicators before returning error
                progress_bar.empty()
                status_text.empty()
                return f"Error: {str(e)}"
    
    # Clear progress indicators before final return
    progress_bar.empty()
    status_text.empty()
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
    
    # Extract and clean texts with metadata
    texts = []
    for _, row in domain_df.iterrows():
        # Clean the main text
        cleaned_text = clean_text_for_theme_analysis(row['Reason_W_AddDetails'])
        if cleaned_text:  # Only add non-empty texts
            # Find the account name column
            account_col = next((col for col in row.index if 'Account Name' in col), None)
            account_name = row[account_col] if account_col else 'Unknown Account'
            
            # Get product information
            product = row.get('Product', 'Unknown Product')
            
            created_date = row.get('Created Date', '')
            if created_date:
                try:
                    created_date = pd.to_datetime(created_date).strftime('%Y-%m-%d')
                except:
                    created_date = str(created_date)
            
            formatted_text = f"{cleaned_text} [METADATA]Created: {created_date}, Account: {account_name}, **Product**: {product}[/METADATA]"
            texts.append(formatted_text)
    
    return texts

def group_similar_requests(texts, embedding_model, domain=None, similarity_threshold=0.85):
    """Group similar feature requests together using embeddings.
    
    Args:
        texts (list): List of feature request texts
        embedding_model: The SentenceTransformer model for generating embeddings
        domain (str, optional): Domain name to check for cached embeddings
        similarity_threshold (float): Threshold for considering texts similar
        
    Returns:
        list: List of groups, where each group is a dict containing:
            - texts: List of all texts in the group
            - size: Total number of texts in the group
            - examples: Up to 5 representative examples
    """
    if not texts:
        return []
    
    # Try to use cached embeddings if available
    if domain and domain in st.session_state.domain_embeddings:
        cached = st.session_state.domain_embeddings[domain]
        cached_texts = set(cached['texts'])
        current_texts = set(texts)
        
        # Check if all current texts are in cached texts
        if current_texts.issubset(cached_texts):
            # Get indices of current texts in cached texts
            text_to_idx = {text: idx for idx, text in enumerate(cached['texts'])}
            indices = [text_to_idx[text] for text in texts]
            embeddings = cached['embeddings'][indices]
            logging.info(f"Using cached embeddings for domain {domain} - {len(texts)} texts")
        else:
            # Generate new embeddings if texts don't match
            logging.info(f"Cache mismatch for domain {domain} - generating new embeddings for {len(texts)} texts")
            embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    else:
        # Generate new embeddings if no cache available
        if domain:
            logging.info(f"No cache found for domain {domain} - generating new embeddings for {len(texts)} texts")
        else:
            logging.info(f"No domain specified - generating new embeddings for {len(texts)} texts")
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Initialize groups
    grouped = []
    used_indices = set()
    
    # Group similar texts
    for i in range(len(texts)):
        if i in used_indices:
            continue
            
        # Find similar texts
        group = [i]
        used_indices.add(i)
        
        for j in range(i + 1, len(texts)):
            if j not in used_indices and similarity_matrix[i, j] > similarity_threshold:
                group.append(j)
                used_indices.add(j)
        
        if group:
            group_texts = [texts[idx] for idx in group]
            
            # Select representative examples
            # If we have 5 or fewer texts, use all of them
            # If we have more than 5, select 5 representative ones:
            # - First text (the "seed" text that started the group)
            # - Last text (often most different from the seed)
            # - Three evenly spaced texts from the middle
            if len(group_texts) <= 5:
                examples = group_texts
            else:
                # Always include first and last
                examples = [group_texts[0], group_texts[-1]]
                
                # Add exactly three evenly spaced examples from the middle
                remaining_length = len(group_texts) - 2  # Length excluding first and last
                if remaining_length >= 3:
                    # Calculate indices for three evenly spaced points
                    middle_indices = []
                    for i in range(3):
                        # This ensures even spacing between first and last indices
                        # We want points at 1/4, 1/2, and 3/4 of the way through
                        idx = 1 + int((remaining_length - 1) * ((i + 1) / 4))
                        if idx not in middle_indices:  # Avoid duplicates
                            middle_indices.append(idx)
                    
                    # If we somehow got fewer than 3 points, add more
                    while len(middle_indices) < 3:
                        for i in range(1, remaining_length):
                            if i not in middle_indices and i not in [0, remaining_length-1]:
                                middle_indices.append(i)
                                if len(middle_indices) == 3:
                                    break
                    
                    # Sort middle indices to maintain order
                    middle_indices.sort()
                    
                    # Add the middle examples
                    for idx in middle_indices:
                        examples.append(group_texts[idx])
                    
                # Sort examples to maintain original order
                examples.sort(key=lambda x: group_texts.index(x))
                
                # Verify we have exactly 5 examples
                assert len(examples) == 5, f"Expected 5 examples, got {len(examples)}"
            
            grouped.append({
                "texts": group_texts,
                "size": len(group_texts),
                "examples": examples
            })
    
    # Sort groups by size
    grouped.sort(key=lambda x: x["size"], reverse=True)
    logging.info(f"Created {len(grouped)} groups from {len(texts)} texts")
    
    return grouped

def analyze_themes_with_llm(texts, domain, top_n, llm, embedding_model=None):
    """Analyze texts using LLM to extract themes"""
    # Limit number of examples to prevent token overflow
    max_texts = 150
    total_texts = len(texts)
    
    # If we have more texts than the limit, take a representative sample
    if total_texts > max_texts:
        selected_texts = texts[:max_texts//2] + texts[-max_texts//2:]
        text_note = f"\n\nNote: Analysis based on a representative sample of {max_texts} entries from {total_texts} total entries."
    else:
        selected_texts = texts
        text_note = ""

    # Group similar feature requests if embedding model is provided
    feature_groups = []
    ungrouped_texts = selected_texts.copy()
    feature_groups_section = ""
    total_feature_requests = 0
    
    if embedding_model:
        # Filter texts that are likely feature requests
        feature_texts = [
            text for text in selected_texts 
            if any(keyword in text.lower() for keyword in ['feature', 'enhancement', 'request', 'new'])
        ]
        total_feature_requests = len(feature_texts)
        
        if feature_texts:
            feature_groups = group_similar_requests(feature_texts, embedding_model, domain)
            # Remove grouped texts from ungrouped list
            grouped_texts = set()
            for group in feature_groups:
                grouped_texts.update(group['texts'])
            ungrouped_texts = [text for text in ungrouped_texts if text not in grouped_texts]

    # Prepare feature groups section if we have groups
    if feature_groups:
        feature_groups_section = f"\n\nFEATURE REQUEST STATISTICS:\n"
        feature_groups_section += f"• Total Feature Requests: {total_feature_requests}\n"
        feature_groups_section += f"• Number of Groups: {len(feature_groups)}\n"
        feature_groups_section += f"• Grouped Requests: {sum(group['size'] for group in feature_groups)}\n"
        feature_groups_section += f"• Ungrouped Requests: {len(ungrouped_texts)}\n\n"
        
        feature_groups_section += "For Feature Enhancement themes, similar requests are grouped together. For each group:\n"
        feature_groups_section += "1. Group theme/category name based on common elements\n"
        feature_groups_section += "2. Number of requests in the group (total count)\n"
        feature_groups_section += "3. Common elements or patterns shared across the group\n"
        feature_groups_section += "4. Representative examples:\n"
        feature_groups_section += "   - If group has 5 or more requests: exactly 5 representative examples\n"
        feature_groups_section += "   - If group has fewer than 5 requests: all requests in the group\n"
        
        # Add the actual groups data
        feature_groups_section += "\n\nFeature Request Groups:\n"
        for i, group in enumerate(feature_groups, 1):
            feature_groups_section += f"\nGroup {i}:\n"
            feature_groups_section += f"Total Requests: {group['size']}\n"
            feature_groups_section += "Representative Examples:\n"
            for example in group['examples']:
                feature_groups_section += f"- {example}\n"
        
        # Add ungrouped items section if there are any
        if ungrouped_texts:
            feature_groups_section += "\n\n### Ungrouped Feature Requests\n"
            feature_groups_section += f"The following {len(ungrouped_texts)} items could not be grouped with others due to low similarity:\n\n"
            # Show up to 8 examples with metadata
            for example in ungrouped_texts[:8]:
                # Extract metadata from the example text
                metadata_start = example.find("[METADATA]")
                metadata_end = example.find("[/METADATA]")
                if metadata_start != -1 and metadata_end != -1:
                    text = example[:metadata_start].strip()
                    metadata = example[metadata_start+9:metadata_end].strip()
                    # Format the example with metadata
                    feature_groups_section += f"- {text} ({metadata})\n"
                else:
                    feature_groups_section += f"- {example}\n"
            if len(ungrouped_texts) > 8:
                feature_groups_section += f"\n... and {len(ungrouped_texts) - 8} more items\n"

    prompt = f"""Analyze the following text entries from the {domain} solution domain and identify EXACTLY {top_n} complaint themes.

IMPORTANT INSTRUCTIONS:
1. You MUST generate EXACTLY {top_n} distinct themes, no more and no less.
2. Feature Enhancement Requests MUST be Theme #1 if present in the data.
3. Other themes should be distinct from Feature Enhancements.
4. Each theme MUST have its own number and complete analysis.
5. Do not combine or merge themes - keep them separate.
6. Follow the exact format specified below.
7. For Feature Enhancement Requests theme:
   - Use EXACTLY the counts provided in the FEATURE REQUEST STATISTICS section
   - Use EXACTLY the groups provided in the Feature Request Groups section
   - Do not create new groups or modify existing ones
   - Report the exact total count as provided in the statistics

For each theme, provide:
1. Clear title and occurrence count (use exact counts for Feature Requests)
2. Detailed description paragraph (2-3 sentences)
3. Examples from the dataset, grouped by similarity for feature requests, including any ungrouped items
4. Supporting evidence and impact analysis
5. Actionable recommendations{feature_groups_section}

Your response MUST follow this exact format and MUST be complete. For feature request themes:
1. First show the grouped examples in their respective groups
2. Then ALWAYS show the "Ungrouped Feature Requests" section if there are any ungrouped items
3. Format each ungrouped item as: "- **[Customer]** – [Details] (Created: [Date], Account: [Name], **Product**: [Product])"
4. Show up to 8 ungrouped examples
5. If there are more than 8 ungrouped items, add a line showing the count of remaining items

# **Analysis of Top {top_n} Complaint Themes in Solutions Domain {domain}**

[For each theme:]
## **[Number]. [Theme Title]**  
**Occurrences**: [EXACT total count of ALL matching examples]

[1-2 sentence description]

### **Examples:**  
[If theme is about feature requests, for each group show:]
#### Group [Number]: [Group Theme/Category]
**Count**: [Number of requests in group]
**Common Elements**: [What makes these requests similar]
**Representative Examples**:
[Show ALL examples if group has fewer than 5 requests, or EXACTLY 5 representative examples if group has 5 or more]
- **[Name]** – [Details] (Created: [Date], Account: [Name], **Product**: [Product])

[For non-feature request themes, show examples as:]
- **[Customer]** – [Details] (Created: [Date], Account: [Name], **Product**: [Product])

### **Impact:**
• Business: [Impact]
• Customer: [Impact]
• Operations: [Impact]

### **Recommendations:**
1. [Short-term]
2. [Medium-term]

# **Comprehensive Impact & Recommendations Summary**

## **Consolidated Business Impact Analysis**
[Summarize all business impacts across themes, highlighting patterns and critical areas]

## **Customer Experience Impact Overview**
[Summarize all customer impacts, emphasizing recurring pain points and satisfaction blockers]

## **Operational Impact Assessment**
[Summarize all operational impacts, focusing on efficiency and resource implications]

## **Strategic Recommendations**

### **Immediate Actions (0-3 months)**
[Prioritized list combining and consolidating all short-term recommendations across themes]
1. [High Priority Action]
2. [Medium Priority Action]
3. [Standard Priority Action]

### **Medium-Term Initiatives (3-6 months)**
[Strategic initiatives combining medium-term recommendations across themes]
1. [Strategic Initiative]
2. [Process Improvement]
3. [System Enhancement]

### **Implementation Guidance**
• Resource Requirements: [Key resources needed]
• Dependencies: [Critical dependencies]
• Success Metrics: [Key performance indicators]

Analyze:
{selected_texts}{text_note}"""

    try:
        # Call LLM with retry mechanism
        for attempt in range(CONFIG['retry']['max_retries']):
            try:
                response = llm.invoke(prompt)
                if response and isinstance(response.content, str):
                    # Verify the response is complete
                    if "### **Implementation Guidance**" not in response.content:
                        if attempt == CONFIG['retry']['max_retries'] - 1:
                            if top_n > 3:
                                raise Exception("Report generation incomplete. Please try reducing the number of themes to 3 or less.")
                            else:
                                raise Exception("Report generation incomplete. Please try with a smaller date range.")
                        continue
                    return response.content
            except Exception as e:
                if attempt == CONFIG['retry']['max_retries'] - 1:
                    if "maximum context length" in str(e).lower():
                        raise Exception("Too much data to analyze. Please try with a smaller date range or fewer themes.")
                    raise e
                time.sleep(CONFIG['retry']['sleep_seconds'])
        
        return None
    except Exception as e:
        logger.error(f"Error in theme analysis: {str(e)}")
        raise Exception(f"Report generation error: {str(e)}")

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

def get_file_info(file_path, file_manager):
    """Get file information including size, last modified time, and hash"""
    try:
        file_stat = os.stat(file_path)
        metadata = file_manager.get_file_metadata(os.path.basename(file_path))
        return {
            'size': file_stat.st_size,
            'last_modified': datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'hash': metadata.get('hash', 'N/A') if metadata else 'N/A'
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {str(e)}")
        return None

def handle_file_management():
    """Handle file management tab functionality"""
    st.markdown("### 📂 File Management")
    
    if not st.session_state.file_manager:
        st.error("File manager not initialized. Please try logging in again.")
        return
        
    # Get list of files in uploads directory
    uploads_dir = st.session_state.file_manager.uploads_dir
    if not uploads_dir.exists():
        st.warning("No uploaded files found.")
        return
        
    files = list(uploads_dir.glob('*'))
    if not files:
        st.info("No files in uploads directory.")
        return
        
    # Create a list of file information
    file_info = []
    total_size = 0
    for file_path in files:
        info = get_file_info(file_path, st.session_state.file_manager)
        if info:
            info['name'] = file_path.name
            file_info.append(info)
            total_size += info['size']
    
    # Display storage usage
    st.markdown("#### 💾 Storage Usage")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Files", len(files))
    with col2:
        st.metric("Total Size", f"{total_size / (1024*1024):.2f} MB")
    
    # Create a dataframe for file information
    if file_info:
        df_files = pd.DataFrame(file_info)
        df_files['size'] = df_files['size'].apply(lambda x: f"{x / (1024*1024):.2f} MB")
        df_files = df_files.rename(columns={
            'name': 'File Name',
            'size': 'Size',
            'last_modified': 'Last Modified',
            'hash': 'MD5 Hash'
        })
        
        st.markdown("#### 📄 Uploaded Files")
        st.dataframe(df_files, use_container_width=True)
        
        # File deletion section
        st.markdown("#### 🗑️ File Management")
        
        # Single file deletion
        file_to_delete = st.selectbox(
            "Select file to delete",
            options=df_files['File Name'].tolist(),
            help="Select a file to delete"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Delete Selected File", type="secondary"):
                try:
                    # Use FileManager's delete_file method
                    st.session_state.file_manager.delete_file(file_to_delete)
                    st.success(f"Successfully deleted {file_to_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting file: {str(e)}")
        
        with col2:
            if st.button("🗑️ Delete All Files", type="secondary"):
                try:
                    for file_path in files:
                        # Use FileManager's delete_file method for each file
                        st.session_state.file_manager.delete_file(file_path.name)
                    st.success("Successfully deleted all files")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting files: {str(e)}")

def cleanup_old_files(file_manager, keep_latest=True):
    """Clean up old files in the uploads directory"""
    try:
        uploads_dir = file_manager.uploads_dir
        if not uploads_dir.exists():
            return
            
        files = list(uploads_dir.glob('*'))
        if not files:
            return
            
        # If keeping latest, sort by modification time and remove all but the newest
        if keep_latest:
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            files_to_delete = files[1:]  # Keep the most recent file
        else:
            files_to_delete = files  # Delete all files
            
        # Delete files
        for file_path in files_to_delete:
            try:
                # Use FileManager's delete_file method
                file_manager.delete_file(file_path.name)
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
    except Exception as e:
        logger.error(f"Error in cleanup_old_files: {str(e)}")

def process_all_domains(df, embedding_model, classifier):
    """Process all domains in the DataFrame"""
    try:
        # Initialize progress tracking
        main_progress = st.progress(0)
        main_status = st.empty()
        main_status.text("Starting processing...")
        
        # Initialize required columns
        required_columns = ['Solution Domain', 'possibleDuplicates', 'CrossDomainDuplicates', 'Sentiment']
        
        # Process all domains
        all_domains_df = pd.DataFrame()
        all_texts = []
        text_to_row_map = {}
        
        # Create progress tracking for multiple domains
        domains_progress = st.progress(0)
        domains_status = st.empty()
        
        # Track processed domains count for progress
        processed_count = 0
        unique_domains = df['Solution Domain'].unique().tolist()
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
                
                # Initialize required columns if they don't exist
                for col in required_columns:
                    if col not in domain_df.columns:
                        domain_df[col] = ""
                
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
            threshold = CONFIG['duplicates']['similarity_threshold']
            for i in range(len(all_texts)):
                for j in range(i):
                    if similarity_matrix[i, j] > threshold:
                        duplicate_info = text_to_row_map[j]
                        current_info = text_to_row_map[i]
                        
                        # Create duplicate messages
                        if duplicate_info['domain'] != current_info['domain']:
                            # Cross-domain duplicate
                            all_domains_df.at[current_info['df_index'], 'CrossDomainDuplicates'] = f"Duplicate of Row {duplicate_info['original_row']} in domain '{duplicate_info['domain']}'"
                            all_domains_df.at[duplicate_info['df_index'], 'CrossDomainDuplicates'] = f"Has duplicates in domain '{current_info['domain']}' (Row {current_info['original_row']})"
                        else:
                            # Within-domain duplicate
                            all_domains_df.at[current_info['df_index'], 'possibleDuplicates'] = f"Duplicate of Row {duplicate_info['original_row']}"
                            all_domains_df.at[duplicate_info['df_index'], 'possibleDuplicates'] = f"Has duplicates (Row {current_info['original_row']})"
        
        # Save processed data
        save_processed_data(all_domains_df)
        
        # Store results in session state
        st.session_state.processed_df = all_domains_df
        st.session_state.processed_df_columns = list(all_domains_df.columns)
        st.session_state.selected_domain = "ALL Domains"
        
        # Clean up progress indicators
        main_progress.empty()
        main_status.empty()
        domains_progress.empty()
        domains_status.empty()
        
        return all_domains_df
        
    except Exception as e:
        # Clean up progress indicators
        if 'main_progress' in locals():
            main_progress.empty()
        if 'main_status' in locals():
            main_status.empty()
        if 'domains_progress' in locals():
            domains_progress.empty()
        if 'domains_status' in locals():
            domains_status.empty()
        raise e

def process_uploaded_file(uploaded_file):
    """Process the uploaded file and update session state"""
    try:
        # First save the file using FileManager
        if not st.session_state.file_manager:
            st.error("File manager not initialized. Please try logging in again.")
            return
            
        # Save the file and get the saved path
        saved_path = st.session_state.file_manager.save_file(uploaded_file, uploaded_file.name)
        if not saved_path:
            st.error("Failed to save uploaded file")
            return
            
        # Load Excel file for sheet selection
        xl = pd.ExcelFile(saved_path)
        sheet_name = st.sidebar.selectbox(
            "Select Sheet",
            options=xl.sheet_names,
            format_func=lambda x: f"Sheet: {x}",
            key="process_file_sheet_selector"
        )
        
        if sheet_name:
            # Load selected sheet
            df = pd.read_excel(saved_path, sheet_name=sheet_name)
            
            # Fill empty cells with values from above, handling consecutive empty cells
            columns_to_fill = ['Account Name', 'Product', 'Use Case']
            for col in columns_to_fill:
                matching_cols = [c for c in df.columns if col in c]
                if matching_cols:
                    # Replace 'None' strings with NaN
                    df[matching_cols[0]] = df[matching_cols[0]].replace('None', pd.NA)
                    # Convert empty strings to NaN
                    df[matching_cols[0]] = df[matching_cols[0]].replace(r'^\s*$', pd.NA, regex=True)
                    
                    # Get the first non-empty value
                    first_valid = df[matching_cols[0]].first_valid_index()
                    if first_valid is not None:
                        first_value = df.at[first_valid, matching_cols[0]]
                        # Fill NaN values before first valid value
                        df.loc[:first_valid, matching_cols[0]] = df.loc[:first_valid, matching_cols[0]].fillna(first_value)
                    
                    # Forward fill the rest
                    df[matching_cols[0]] = df[matching_cols[0]].fillna(method='ffill')
                    
                    # Handle any remaining NaN values at the end
                    last_valid = df[matching_cols[0]].last_valid_index()
                    if last_valid is not None:
                        last_value = df.at[last_valid, matching_cols[0]]
                        df.loc[last_valid:, matching_cols[0]] = df.loc[last_valid:, matching_cols[0]].fillna(last_value)
            
            # Time period selection
            if 'Created Date' in df.columns:
                df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
                df = df[~df['Created Date'].isna()].copy()
                
                if len(df) > 0:
                    min_date = df['Created Date'].min().date()
                    max_date = df['Created Date'].max().date()
                    
                    time_period = st.sidebar.radio(
                        "Select Time Period",
                        ["Entire Time Period", "Custom Time Period"],
                        key="process_file_time_period"
                    )
                    
                    if time_period == "Custom Time Period":
                        col1, col2 = st.sidebar.columns(2)
                        with col1:
                            start_date = st.date_input(
                                "Start Date",
                                min_value=min_date,
                                max_value=max_date,
                                value=min_date,
                                key="start_date"
                            )
                        with col2:
                            end_date = st.date_input(
                                "End Date",
                                min_value=min_date,
                                max_value=max_date,
                                value=max_date,
                                key="end_date"
                            )
                            
                            if start_date > end_date:
                                st.sidebar.error("End date must be after start date")
                                return
                    
                    # Add Process button after time period selection
                    st.sidebar.markdown("---")  # Add a separator
                    if st.sidebar.button("Process Data", type="primary", key="process_data_uploaded"):
                        # Process the filtered data
                        with st.spinner('Processing data...'):
                            embedding_model, classifier = load_models()
                            all_domains_df = process_all_domains(df, embedding_model, classifier)
                            
                            if all_domains_df is not None:
                                st.session_state.processed_df = all_domains_df
                                st.session_state.selected_domain = "ALL Domains"
                                st.success("✅ File processed successfully!")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        logger.error(f"File processing error: {str(e)}", exc_info=True)

def display_sidebar_info():
    """Display information about models and system in the sidebar"""
    st.sidebar.markdown("### Upload Data")
    
    # Current User
    if st.session_state.current_user:
        st.sidebar.markdown(f"👤 Current User: {st.session_state.current_user}")
    
    # Upload Excel File section
    uploaded_file = st.sidebar.file_uploader(
        "Upload Excel File (up to 1GB)",
        type=['xlsx'],
        help="Limit 200MB per file • XLSX"
    )
    
    # Add note about uploading Excel file
    st.sidebar.info("📝 Note: Upload an Excel file to process new data.")

    # Process uploaded file if present
    if uploaded_file:
        process_uploaded_file(uploaded_file)
    
    # Models Used section
    st.sidebar.markdown("### 🎮 Models Used")
    
    # Local Models section
    st.sidebar.markdown("Local Models")
    
    # Sentiment Analysis expander
    with st.sidebar.expander("Sentiment Analysis", expanded=False):
        st.markdown("""
        - **Model**: facebook/bart-large-mnli
        - **Type**: Zero-shot classifier
        - **Usage**: Sentiment & importance rating
        - ✅ Runs completely offline
        """)
    
    # Text Embeddings expander
    with st.sidebar.expander("Text Embeddings", expanded=False):
        st.markdown("""
        - **Model**: all-MiniLM-L6-v2
        - **Type**: SentenceTransformer
        - **Usage**: Duplicate detection
        - ✅ Fully offline processing
        - 384-dimensional embeddings
        """)
    
    # Secure Search section
    st.sidebar.markdown("Secure Search")
    
    # Fuzzy Search expander
    with st.sidebar.expander("Fuzzy Search", expanded=False):
        st.markdown("""
        - **Framework**: Langchain
        - **Model**: BridgeIT (Cisco Internal)
        - ✅ Secure enterprise authentication
        - ✅ Enterprise-grade security
        - ✅ Advanced natural language processing
        """)
    
    # Device information in info box
    device = "Apple M-series GPU (MPS)" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else \
             "NVIDIA GPU (CUDA)" if torch.cuda.is_available() else \
             "CPU"
    st.sidebar.info(f"Using device: {device} (MPS)")

def validate_and_load_df():
    """Validate and load the processed DataFrame"""
    if 'processed_df' not in st.session_state or st.session_state.processed_df is None:
        st.warning("Please upload and process a file first.")
        return False
    return True

def display_main_analysis_tab():
    """Display main analysis tab content"""
    st.markdown("## 📊 Main Analysis")
    
    if st.session_state.processed_df is not None:
        # Get unique domains for selection
        domains = ["ALL Domains"] + sorted(st.session_state.processed_df['Solution Domain'].unique().tolist())
        
        # Domain selection
        selected_domain = st.selectbox(
            "Select Solution Domain",
            options=domains,
            index=domains.index(st.session_state.selected_domain) if st.session_state.selected_domain in domains else 0
        )
        
        # Filter data based on selected domain
        if selected_domain == "ALL Domains":
            df_display = st.session_state.processed_df.copy()
        else:
            df_display = st.session_state.processed_df[
                st.session_state.processed_df['Solution Domain'] == selected_domain
            ].copy()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df_display))
        with col2:
            st.metric("Unique Solutions", df_display['Solution Domain'].nunique())
        with col3:
            duplicate_count = len(df_display[
                (df_display['possibleDuplicates'].str.len() > 0) | 
                (df_display['CrossDomainDuplicates'].str.len() > 0)
            ])
            st.metric("Potential Duplicates", duplicate_count)
        
        # Display main DataFrame
        st.markdown("### Complete Dataset")
        st.dataframe(df_display, use_container_width=True)
        
        # Display duplicates DataFrame if there are duplicates
        duplicates_mask = (
            (df_display['possibleDuplicates'].str.len() > 0) |
            (df_display['CrossDomainDuplicates'].str.len() > 0)
        )
        duplicates_df = df_display[duplicates_mask].copy()
        
        if len(duplicates_df) > 0:
            st.markdown("### 🔄 Duplicate Entries")
            st.dataframe(duplicates_df, use_container_width=True)
        
        # Download options for main DataFrame
        col1, col2 = st.columns(2)
        with col1:
            # Excel download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_display.to_excel(writer, index=False)
            st.download_button(
                label="📥 Download Complete Dataset as Excel",
                data=buffer.getvalue(),
                file_name=f"analysis_results_{selected_domain.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # CSV download
            csv = df_display.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Dataset as CSV",
                data=csv,
                file_name=f"analysis_results_{selected_domain.replace(' ', '_')}.csv",
                mime="text/csv"
            )
        
        # Display domain distribution summary
        st.markdown("### 📊 Domain Distribution Summary")
        domain_counts = df_display['Solution Domain'].value_counts()
        for domain, count in domain_counts.items():
            st.text(f"{domain}: {count} entries")
        
        # Display duplicate summary if there are duplicates
        if len(duplicates_df) > 0:
            st.markdown("### 📊 Duplicate Summary")
            domain_dup_counts = duplicates_df['Solution Domain'].value_counts()
            for domain, count in domain_dup_counts.items():
                st.text(f"{domain}: {count} duplicates")
            
            # Add download options for duplicates DataFrame
            st.markdown("### Download Duplicates Data")
            dup_col1, dup_col2 = st.columns(2)
            with dup_col1:
                # Excel download for duplicates
                dup_buffer = io.BytesIO()
                with pd.ExcelWriter(dup_buffer, engine='openpyxl') as writer:
                    duplicates_df.to_excel(writer, index=False)
                st.download_button(
                    label="📥 Download Duplicates as Excel",
                    data=dup_buffer.getvalue(),
                    file_name=f"duplicates_{selected_domain.replace(' ', '_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with dup_col2:
                # CSV download for duplicates
                dup_csv = duplicates_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Duplicates as CSV",
                    data=dup_csv,
                    file_name=f"duplicates_{selected_domain.replace(' ', '_')}.csv",
                    mime="text/csv"
                )

def display_fuzzy_search_tab():
    """Display fuzzy search tab content"""
    st.markdown("## 🔍 Fuzzy Search")
    
    if st.session_state.processed_df is not None:
        # Example queries in collapsible box
        with st.expander("📝 Example Queries"):
            st.markdown("""
            - Show me ALL rows that have Negative sentiment across ALL solution domains
            - Show me ALL rows that have Negative sentiment for domain data center
            - Show me the records between March 15th 2024 and March 16th 2024
            - Show me all rows for possible duplicates for data center networking
            - Show me the rows where the account name matches 'BU - MAN DE'
            - Show me the rows where the account name has at&t and sentiment is negative
            - Show me all rows that are duplicates and also solutions domain has campus and have negative sentiment and created in April 2024
            - Show me all rows that have highrating
            - Show me all rows that have highrating+
            - show me all the rows that have exactly highrating
            - show me all the rows that have exactly highrating and also duplicates 
            - Show me the records that have cross domain duplicates  and sentiment has negative                      
            - Show me the records that have possible duplicates  and sentiment has negative  and contains complex
            """)
        
        # Create the fuzzy search interface
        query = st.text_area(
            "Enter your search query:",
            height=100,
            placeholder="e.g., Show me all rows that have Negative sentiment",
            key="fuzzy_search_query"
        )
        
        # Single container for all status and results
        result_container = st.container()
        
        if st.button("🔍 Search", type="primary", key="fuzzy_search_button"):
            if query:
                try:
                    with result_container:
                        with st.spinner('Processing your query...'):
                            # Initialize LLM and run search in one step
                            try:
                                llm = init_azure_openai()
                                result = run_fuzzy_search_query_with_retry(
                                    query, 
                                    st.session_state.processed_df, 
                                    llm
                                )
                            except Exception as e:
                                st.error(f"Error processing query: {str(e)}")
                                return
                            
                            # Display results
                            if result == "No matching results found.":
                                st.warning("No results found for your query")
                            elif result.startswith("Error:"):
                                st.error(result)
                            else:
                                df_result, error = parse_markdown_table(result)
                                if df_result is not None:
                                    display_fuzzy_search_results(df_result)
                                else:
                                    st.warning(error or "No results found for your query")
                                    
                except Exception as e:
                    with result_container:
                        st.error(f"Error processing query: {str(e)}")
            else:
                with result_container:
                    st.warning("Please enter a search query")

def display_reports_tab():
    """Display reports tab content"""
    st.markdown("## 📝 Theme Analysis Reports")
    
    try:
        if st.session_state.processed_df is not None:
            # Domain selection
            domains = sorted(st.session_state.processed_df['Solution Domain'].unique().tolist())
            selected_domain = st.selectbox(
                "Select Solution Domain for Analysis",
                options=domains,
                key="report_domain_selector"
            )
            
            # Number of themes selection
            num_themes = st.selectbox(
                "Number of Top Themes to Analyze",
                options=list(range(3, 16)),
                index=1,
                help="Select how many top themes to analyze (between 3 and 10)"
            )
            
            # Initialize report storage in session state if not exists
            if 'current_report' not in st.session_state:
                st.session_state.current_report = None
            if 'current_report_domain' not in st.session_state:
                st.session_state.current_report_domain = None
            
            # Create containers for progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            substep_progress = st.empty()  # New container for substep progress
            
            # Generate report button and processing
            if st.button("Generate Theme Analysis Report", type="primary"):
                try:
                    # Step 1: Data Preparation (0-20%)
                    # 1.1 Initial setup (0-5%)
                    status_text.text("Initializing analysis...")
                    progress_bar.progress(0)
                    substep_progress.text("Setting up analysis environment...")
                    time.sleep(0.5)
                    
                    # 1.2 Extracting texts (5-15%)
                    status_text.text("Preparing data for analysis...")
                    progress_bar.progress(5)
                    substep_progress.text("Extracting and cleaning text data...")
                    domain_texts = extract_domain_texts(st.session_state.processed_df, selected_domain)
                    
                    if not domain_texts:
                        progress_bar.empty()
                        status_text.empty()
                        substep_progress.empty()
                        st.warning("No valid texts found for analysis in the selected domain")
                        return
                    
                    # 1.3 Text preprocessing (15-20%)
                    progress_bar.progress(15)
                    substep_progress.text("Preprocessing text data...")
                    time.sleep(0.5)
                    
                    # Step 2: Model Initialization (20-30%)
                    # 2.1 Loading model (20-25%)
                    status_text.text("Setting up AI model...")
                    progress_bar.progress(20)
                    substep_progress.text("Loading language model...")
                    
                    # 2.2 Model initialization (25-30%)
                    progress_bar.progress(25)
                    substep_progress.text("Initializing AI components...")
                    llm = init_azure_openai()
                    
                    # Step 3: Theme Analysis (30-80%)
                    # 3.1 Initial analysis (30-40%)
                    status_text.text("Analyzing themes...")
                    progress_bar.progress(30)
                    substep_progress.text("Identifying main themes in the data...")
                    
                    # 3.2 Theme processing (40-60%)
                    progress_bar.progress(40)
                    substep_progress.text("Processing theme patterns and relationships...")
                    
                    # 3.3 Example collection (60-70%)
                    progress_bar.progress(60)
                    substep_progress.text("Collecting and organizing examples...")
                    
                    # 3.4 Impact analysis (70-75%)
                    progress_bar.progress(70)
                    substep_progress.text("Analyzing business and customer impact...")
                    
                    # 3.5 Recommendations (75-80%)
                    # Break down recommendations into sub-steps
                    progress_bar.progress(75)
                    substep_progress.text("Analyzing historical patterns for recommendations...")
                    time.sleep(0.2)
                    
                    progress_bar.progress(76)
                    substep_progress.text("Generating short-term recommendations (0-3 months)...")
                    time.sleep(0.2)
                    
                    progress_bar.progress(77)
                    substep_progress.text("Evaluating implementation feasibility...")
                    time.sleep(0.2)
                    
                    progress_bar.progress(78)
                    substep_progress.text("Generating medium-term recommendations (3-6 months)...")
                    time.sleep(0.2)
                    
                    progress_bar.progress(79)
                    substep_progress.text("Prioritizing recommendations by impact...")
                    time.sleep(0.2)
                    
                    progress_bar.progress(80)
                    substep_progress.text("Finalizing recommendation details...")
                    
                    # Generate the actual report
                    report = analyze_themes_with_llm(domain_texts, selected_domain, num_themes, llm)
                    
                    if report:
                        # Step 4: Report Finalization (80-100%)
                        # 4.1 Format checking (80-85%)
                        status_text.text("Finalizing report...")
                        progress_bar.progress(80)
                        substep_progress.text("Verifying report format...")
                        
                        # 4.2 Content validation (85-90%)
                        progress_bar.progress(85)
                        substep_progress.text("Validating report content...")
                        
                        # 4.3 Final formatting (90-95%)
                        progress_bar.progress(90)
                        substep_progress.text("Applying final formatting...")
                        
                        # 4.4 Saving report (95-100%)
                        progress_bar.progress(95)
                        substep_progress.text("Saving report...")
                        
                        # Store report and domain in session state
                        st.session_state.current_report = report
                        st.session_state.current_report_domain = selected_domain
                        
                        # Complete (100%)
                        progress_bar.progress(100)
                        status_text.text("Report generation complete!")
                        substep_progress.text("✅ Report ready for viewing")
                        time.sleep(1)  # Show completion briefly
                        
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    substep_progress.empty()
                    
                except Exception as e:
                    # Clear progress indicators on error
                    progress_bar.empty()
                    status_text.empty()
                    substep_progress.empty()
                    st.error(f"Error generating report: {str(e)}")
                    return
            
            # Display current report if it exists
            if st.session_state.current_report:
                st.markdown("### Current Report")
                if st.session_state.current_report_domain:
                    st.info(f"Report for domain: {st.session_state.current_report_domain}")
                
                # Display report
                st.markdown(st.session_state.current_report)
                
                # Show report statistics
                show_report_stats(st.session_state.current_report)
                
                # Download options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Markdown download
                    st.download_button(
                        "📥 Download as Markdown",
                        st.session_state.current_report,
                        file_name=f"theme_analysis_{st.session_state.current_report_domain.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                
                with col2:
                    # PDF download
                    try:
                        pdf_report = create_pdf_report(st.session_state.current_report, st.session_state.current_report_domain)
                        st.download_button(
                            "📥 Download as PDF",
                            pdf_report,
                            file_name=f"theme_analysis_{st.session_state.current_report_domain.replace(' ', '_')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error creating PDF: {str(e)}")
                
                # Add a clear report button
                if st.button("🗑️ Clear Current Report", type="secondary"):
                    st.session_state.current_report = None
                    st.session_state.current_report_domain = None
                    st.rerun()
    
    except Exception as e:
        st.error(f"Error in reports tab: {str(e)}")

def main():
    # Set page config
    st.set_page_config(
        page_title="Solutions Domain Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Show login page first
    if not login_page():
        return
    
    # Initialize managers for current user
    initialize_user_managers()
    
    # Initialize session state for data persistence
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'selected_domain' not in st.session_state:
        st.session_state.selected_domain = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'fuzzy_search_df' not in st.session_state:
        st.session_state.fuzzy_search_df = None
    if 'current_report' not in st.session_state:
        st.session_state.current_report = None
    if 'domain_embeddings' not in st.session_state:
        st.session_state.domain_embeddings = {}
    
    # Create tabs for different functionalities with state management
    tab_names = ["📊 Main Analysis", "🔍 Fuzzy Search", "📝 Reports", "📂 File Management"]
    main_tab, fuzzy_search_tab, reports_tab, file_management_tab = st.tabs(tab_names)
    
    # Store tabs in session state
    st.session_state.main_tab = main_tab
    st.session_state.fuzzy_search_tab = fuzzy_search_tab
    st.session_state.reports_tab = reports_tab
    st.session_state.file_management_tab = file_management_tab
    
    # Display sidebar content
    display_sidebar_info()
    
    # Process uploaded file if present
    if st.session_state.uploaded_file:
        process_uploaded_file(st.session_state.uploaded_file)
    
    # Handle file management tab separately since it doesn't depend on processed_df
    with file_management_tab:
        handle_file_management()
    
    # Only show data-dependent tabs if we have processed data
    if validate_and_load_df():
        # Display content in each tab
        with main_tab:
            display_main_analysis_tab()
        
        with fuzzy_search_tab:
            display_fuzzy_search_tab()
        
        with reports_tab:
            display_reports_tab()

if __name__ == "__main__":
    main() 