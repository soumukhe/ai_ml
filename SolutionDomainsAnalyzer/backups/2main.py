# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import time
from tqdm import tqdm
import re
import torch

import os
#print(os.listdir())

#file_path = 'Adoption Barriers DCNO 20241120(1).xlsx'


# show excel file in data directory
for file in os.listdir('data'):
    if file.endswith('.xlsx'):
        file_path = os.path.join('data', file)
        print(file_path)

# show all sheets
xl = pd.ExcelFile(file_path)
for idx, sheet in enumerate(xl.sheet_names, start=1):
        print(f"{idx}: {sheet}")

# Get user input for sheet selection
while True:
    try:
        sheet_idx = int(input("\nEnter the number of the sheet you want to read: "))
        if 1 <= sheet_idx <= len(xl.sheet_names):
            sheet_name = xl.sheet_names[sheet_idx - 1]
            print(f"Selected sheet: {sheet_name}")
            break
        else:
            print(f"Please enter a number between 1 and {len(xl.sheet_names)}")
    except ValueError:
        print("Please enter a valid number")



# Read the data into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Filter rows where Solution Domain is 'Data Center Networking' and create an explicit copy
data_center_networking = df[df['Solution Domain'] == 'Data Center Networking'].copy()

# Create new column combining Reason and Additional Details using loc
data_center_networking.loc[:, 'Reason_W_AddDetails'] = data_center_networking['Reason'].astype(str) + ' :: ' + data_center_networking['Additional Details'].astype(str)

# Drop the original columns
data_center_networking = data_center_networking.drop(['Reason', 'Additional Details'], axis=1)

# Display the first few rows of the filtered DataFrame
print("\nShape of filtered data (rows, columns):", data_center_networking.shape)

# print(data_center_networking['Reason_W_AddDetails'])



def clean_text(text):
    """Clean and validate text for API submission"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    # Remove special characters and excessive whitespace
    text = re.sub(r'[^\w\s.,!?-]', ' ', str(text))
    text = ' '.join(text.split())
    return text

# Apply the function to the 'Reason_W_AddDetails' column
data_center_networking['Reason_W_AddDetails'] = data_center_networking['Reason_W_AddDetails'].apply(clean_text)

print(data_center_networking['Reason_W_AddDetails'])


# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Set to CPU mode
model.to('cpu')

def get_embeddings_batch(texts, batch_size=32):
    """Get embeddings for a batch of texts using Hugging Face model"""
    all_embeddings = []
    total_expected = len(texts)
    
    # Clean all texts first
    cleaned_texts = [clean_text(text) for text in texts]
    
    # Process in batches
    for i in tqdm(range(0, len(cleaned_texts), batch_size), desc="Processing batches"):
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
                print(f"Error in batch {i//batch_size + 1}: {str(e)}")
        
        all_embeddings.extend(batch_embeddings)
    
    # Ensure we have exactly the right number of embeddings
    result = np.array(all_embeddings[:total_expected])
    print(f"\nFinal embeddings shape: {result.shape}")
    return result

# Generate embeddings using the batch processing function
texts = data_center_networking['Reason_W_AddDetails'].tolist()
normalized_embeddings = get_embeddings_batch(texts)

print("\nEmbeddings generated successfully")

# Verify embeddings match the number of texts
if len(normalized_embeddings) != len(texts):
    print(f"Warning: Embeddings size ({len(normalized_embeddings)}) doesn't match texts size ({len(texts)})")
    exit(1)

print("\nComputing similarity matrix...")
# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(normalized_embeddings)
print(f"Similarity matrix shape: {similarity_matrix.shape}")

# Debug: Print sample of similarity matrix
print("\nSample similarity matrix (first 5 rows and columns):")
print(similarity_matrix[:5, :5])

# Set the similarity threshold
threshold = 0.95  # High threshold for stricter duplicate detection
duplicates = [""] * len(texts)

print("\nIdentifying duplicates...")
duplicate_count = 0
for i in tqdm(range(len(texts)), desc="Checking for duplicates"):
    for j in range(i):  # Compare only with previous rows
        if similarity_matrix[i, j] > threshold:
            duplicates[i] = f"Duplicate of Row {j + 1}"
            duplicate_count += 1
            # Debug: Print the duplicate pairs and their similarity score
            print(f"\nPotential duplicate found:")
            print(f"Row {i + 1} appears to be a duplicate of Row {j + 1}")
            print(f"Similarity score: {similarity_matrix[i, j]:.4f}")
            print(f"Original text:")
            print(f"Text 1: {texts[i]}")
            print(f"Text 2: {texts[j]}")
            break

print(f"\nTotal duplicates found: {duplicate_count}")

# Add duplicates column to the dataframe
data_center_networking['Duplicates'] = duplicates

# Display sample of results
print("\nSample of results with duplicates:")
print(data_center_networking[['Reason_W_AddDetails', 'Duplicates']].head())

# Create tempData directory if it doesn't exist
os.makedirs('tempData', exist_ok=True)

# Save the processed DataFrame to Excel
output_path = os.path.join('tempData', 'dcProcessed.xlsx')
data_center_networking.to_excel(output_path, index=False)
print(f"\nProcessed data saved to: {output_path}")