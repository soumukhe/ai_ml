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
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = None

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
    with suppress_stdout_stderr():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_model.to(device)
        
        classifier = pipeline("zero-shot-classification",
                            model="facebook/bart-large-mnli",
                            clean_up_tokenization_spaces=True,
                            multi_label=True,
                            device=device)
    return embedding_model, classifier

def get_embeddings_batch(texts, model, batch_size=32):
    """Get embeddings for a batch of texts using Hugging Face model"""
    all_embeddings = []
    total_expected = len(texts)
    
    # Clean all texts first
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Process in batches
    for i in range(0, len(cleaned_texts), batch_size):
        batch = cleaned_texts[i:i + batch_size]
        # Filter out empty strings but keep track of their positions
        valid_texts = [(idx, text) for idx, text in enumerate(batch) if text.strip()]
        valid_indices = [idx for idx, _ in valid_texts]
        valid_batch = [text for _, text in valid_texts]
        
        # Create zero vectors for this batch
        batch_embeddings = [np.zeros(384) for _ in range(len(batch))]  # MiniLM-L6-v2 has 384 dimensions
        
        if valid_batch:  # Only process if we have valid texts
            try:
                # Get embeddings for valid texts
                embeddings = model.encode(valid_batch, 
                                       normalize_embeddings=True,
                                       show_progress_bar=False)
                
                # Place embeddings in their correct positions
                for emb_idx, valid_idx in enumerate(valid_indices):
                    batch_embeddings[valid_idx] = embeddings[emb_idx]
                
            except Exception as e:
                st.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
        
        all_embeddings.extend(batch_embeddings)
    
    # Ensure we have exactly the right number of embeddings
    result = np.array(all_embeddings[:total_expected])
    return result

def process_domain_data(df, domain, embedding_model, classifier, skip_duplicates=False):
    """Process data for a specific domain"""
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
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
    else:
        # Skip duplicate detection
        progress_bar.progress(70)
    
    # Process sentiments
    status_text.text("Analyzing text sentiment and request importance...")
    progress_bar.progress(80)
    
    # Define sentiment labels
    sentiment_labels = ["highRating+", "highRating", "Neutral", "Negative", "Negative-"]
    
    # Process sentiments
    sentiments = []
    total_texts = len(filtered_domain_df['Reason_W_AddDetails'])
    
    for idx, text in enumerate(filtered_domain_df['Reason_W_AddDetails']):
        try:
            result = classifier(text, 
                             candidate_labels=sentiment_labels,
                             hypothesis_template="This feature request is {}.")
            top_sentiment = result['labels'][0]
            sentiments.append(top_sentiment)
            
            # Update progress for sentiment analysis
            current_progress = 80 + (idx / total_texts * 15)
            progress_bar.progress(int(current_progress))
            
        except Exception as e:
            st.warning(f"Error processing text: {str(e)}")
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

if uploaded_file:
    with st.spinner('Saving file and loading models...'):
        file_path = save_uploaded_file(uploaded_file)
        embedding_model, classifier = load_models()
        
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
            account_col = account_cols[0]  # Use the first column that contains 'Account Name'
            df[account_col] = df[account_col].fillna(method='ffill')
            
            # Show info about filled values
            filled_count = df[account_col].notna().sum() - df[account_col].count()
            if filled_count > 0:
                st.info(f"Filled {filled_count} empty Account Name cells with values from rows above")
        
        # Convert Created Date to datetime with robust error handling
        try:
            # First try parsing with specific format M/D/YYYY
            df['Created Date'] = pd.to_datetime(df['Created Date'], format='%m/%d/%Y', errors='coerce')
            
            # If any NaT values, try with other common formats
            if df['Created Date'].isna().any():
                # Try parsing without specific format
                df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
            
            # Filter out rows with invalid dates
            valid_dates_mask = ~df['Created Date'].isna()
            invalid_count = (~valid_dates_mask).sum()
            if invalid_count > 0:
                st.info(f"Filtered out {invalid_count} rows with invalid or missing dates")
            
            df = df[valid_dates_mask].copy()
            
            if len(df) == 0:
                st.error("No valid data found after filtering dates")
                st.stop()
            
            # Get min and max dates from valid dates
            min_date = df['Created Date'].min().date()
            max_date = df['Created Date'].max().date()
            
        except Exception as e:
            st.error(f"""
                Error processing dates in 'Created Date' column.
                Expected format is M/D/YYYY (example: 2/16/2024).
                
                Error details: {str(e)}
            """)
            st.stop()
        
        unique_domains = df['Solution Domain'].unique().tolist()
        
        # Add "ALL Domains" option at the beginning of the list
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
            # Add helper text for date format
            st.sidebar.caption("Note: Date selector shows YYYY/MM/DD format, but dates will be displayed as M/D/YYYY in results")
            
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
            
            # Validate date range
            if start_date > end_date:
                st.sidebar.error("End date must be after start date")
                st.stop()
            
            # Show date range info in M/D/YYYY format
            st.sidebar.info(
                f"Available date range: "
                f"{min_date.strftime('%-m/%-d/%Y')} to "
                f"{max_date.strftime('%-m/%-d/%Y')}"
            )
            
            # Show selected date range in M/D/YYYY format
            st.sidebar.success(
                f"Selected range: "
                f"{start_date.strftime('%-m/%-d/%Y')} to "
                f"{end_date.strftime('%-m/%-d/%Y')}"
            )
        
        if selected_domain and st.sidebar.button("Process Domain"):
            with st.spinner(f'Processing data for domain{"s" if selected_domain == "ALL Domains" else ""}: {selected_domain}...'):
                # Apply date filter if custom time period is selected
                if time_period == "Custom Time Period":
                    # Convert dates to datetime for filtering
                    start_datetime = pd.Timestamp(start_date)
                    end_datetime = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
                    
                    # Filter DataFrame by date range
                    df = df[
                        (df['Created Date'] >= start_datetime) & 
                        (df['Created Date'] <= end_datetime)
                    ]
                    
                    if len(df) == 0:
                        st.error("No data found in the selected date range")
                        st.stop()
                    
                    st.info(f"Filtered to {len(df)} records between {start_date} and {end_date}")
                
                if selected_domain == "ALL Domains":
                    # Process all domains
                    all_domains_df = pd.DataFrame()  # Empty DataFrame to store all results
                    all_texts = []  # Store all texts for cross-domain duplicate detection
                    text_to_row_map = {}  # Map to store original row numbers
                    
                    # Create progress tracking for multiple domains
                    domains_progress = st.progress(0)
                    domains_status = st.empty()
                    
                    # Track processed domains count for progress
                    processed_count = 0
                    total_domains = len(unique_domains)
                    
                    # First pass: Process each domain without duplicate detection
                    for domain in unique_domains:
                        # Check if domain has any data
                        domain_data = df[df['Solution Domain'] == domain]
                        if len(domain_data) == 0:
                            st.warning(f"Skipping empty domain: {domain}")
                            continue
                            
                        try:
                            # Update domain progress
                            progress_percent = (processed_count / total_domains) * 50  # First half of progress
                            domains_progress.progress(int(progress_percent))
                            domains_status.text(f"Processing domain {processed_count + 1} of {total_domains}: {domain}")
                            
                            # Process individual domain (without duplicate detection)
                            domain_df = process_domain_data(df, domain, embedding_model, classifier, skip_duplicates=True)
                            
                            # Add domain identifier column at the start
                            domain_df.insert(0, 'Solution_Domain', domain)
                            
                            # Store texts and row mapping for duplicate detection
                            start_idx = len(all_texts)
                            domain_texts = domain_df['Reason_W_AddDetails'].tolist()
                            all_texts.extend(domain_texts)
                            
                            # Map each text to its domain and row information
                            for i, text in enumerate(domain_texts):
                                text_to_row_map[start_idx + i] = {
                                    'domain': domain,
                                    'original_row': domain_df['Original_Row_Number'].iloc[i],
                                    'df_index': len(all_domains_df) + i
                                }
                            
                            # Append to main DataFrame
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
                        all_domains_df['CrossDomainDuplicates'] = ""  # New column for cross-domain duplicates
                        
                        # Track original rows that have duplicates
                        original_rows = set()
                        
                        # Find duplicates across all domains
                        threshold = 0.95
                        for i in range(len(all_texts)):
                            for j in range(i):
                                if similarity_matrix[i, j] > threshold:
                                    duplicate_info = text_to_row_map[j]
                                    current_info = text_to_row_map[i]
                                    current_idx = current_info['df_index']
                                    original_idx = duplicate_info['df_index']
                                    
                                    # Add original row to tracking set
                                    original_rows.add(original_idx)
                                    
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
                        
                        domains_progress.progress(100)
                        domains_status.empty()
                        domains_progress.empty()
                    
                    if len(all_domains_df) > 0:
                        # Sort the final DataFrame by Solution_Domain
                        all_domains_df = all_domains_df.sort_values('Solution_Domain')
                        st.session_state.processed_df = all_domains_df
                        st.session_state.selected_domain = "ALL Domains"
                        st.success(f"Successfully processed {processed_count} out of {total_domains} domains")
                    else:
                        st.error("No data was processed successfully")
                else:
                    # Process single domain as before
                    st.session_state.processed_df = process_domain_data(
                        df, selected_domain, embedding_model, classifier
                    )
                    st.session_state.selected_domain = selected_domain

# Display results
if st.session_state.processed_df is not None:
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
                styled_filtered = styled_filtered.applymap(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
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
    styled_df = styled_df.applymap(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
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
    duplicates_mask = (
        (st.session_state.processed_df['possibleDuplicates'].str.len() > 0) |
        (st.session_state.processed_df['CrossDomainDuplicates'].str.len() > 0)
    )
    
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
                styled_filtered_dups = styled_filtered_dups.applymap(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
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
        original_entries = duplicates_df[
            (duplicates_df['possibleDuplicates'].str.contains('Has duplicates', na=False)) |
            (duplicates_df['CrossDomainDuplicates'].str.contains('Has duplicates', na=False))
        ]
        duplicate_entries = duplicates_df[
            (duplicates_df['possibleDuplicates'].str.contains('Duplicate of', na=False)) |
            (duplicates_df['CrossDomainDuplicates'].str.contains('Duplicate of', na=False))
        ]
        
        st.markdown(f"""
        Found {len(duplicates_df)} entries in duplicate analysis:
        - {len(original_entries)} original entries that have duplicates
        - {len(duplicate_entries)} duplicate entries referencing other rows
        """)
        
        # Display complete duplicates DataFrame with custom styling
        styled_duplicates = duplicates_df.style.format(na_rep='')
        styled_duplicates = styled_duplicates.applymap(lambda x: 'background-color: #ffeb99' if pd.isna(x) else '')
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

# Footer with copyright
st.markdown("---")
st.markdown("© 2024 Solutions Domain Analyzer") 