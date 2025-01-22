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

@st.cache_resource
def load_models():
    """Load and cache the ML models"""
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

def process_domain_data(df, domain, embedding_model, classifier):
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
    
    # Reorder columns to put Original_Row_Number before possibleDuplicates
    cols = filtered_domain_df.columns.tolist()
    cols.remove('Original_Row_Number')
    cols.remove('possibleDuplicates')
    
    # Reconstruct column order with Original_Row_Number before possibleDuplicates
    new_cols = cols + ['Original_Row_Number', 'possibleDuplicates']
    filtered_domain_df = filtered_domain_df[new_cols]
    
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
        unique_domains = df['Solution Domain'].unique()
        
        selected_domain = st.sidebar.selectbox(
            "Select Solution Domain",
            options=unique_domains,
            format_func=lambda x: f"Domain: {x}"
        )
        
        if selected_domain and st.sidebar.button("Process Domain"):
            with st.spinner(f'Processing data for domain: {selected_domain}...'):
                st.session_state.processed_df = process_domain_data(
                    df, selected_domain, embedding_model, classifier
                )
                st.session_state.selected_domain = selected_domain

# Display results
if st.session_state.processed_df is not None:
    st.markdown(f"### Analysis for Solutions Domain: {st.session_state.selected_domain}")
    st.markdown("---")
    
    # Display the processed DataFrame with custom styling
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

# Footer with copyright
st.markdown("---")
st.markdown("© 2024 Solutions Domain Analyzer") 