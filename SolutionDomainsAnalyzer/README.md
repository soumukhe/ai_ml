# Solutions Domain Analyzer 🔍

A powerful web application that analyzes solutions domain data using advanced Natural Language Processing (NLP) techniques. This tool helps identify duplicate entries (both within and across domains), assess request importance, and analyze sentiment in your solutions domain data.

## Features ✨

- **Interactive Web Interface**: 
  - Clean, professional Streamlit interface
  - Progress tracking with detailed status updates
  - Highlighted visualization of results

- **Advanced NLP Processing**:
  - Text embedding generation using SentenceTransformer
  - Semantic similarity detection (both within and across domains)
  - Zero-shot classification for request importance and sentiment analysis

- **Data Processing**:
  - Automatic duplicate detection with cross-domain support
  - Request importance classification (highRating+, highRating)
  - Sentiment analysis (Neutral, Negative, Negative-)
  - Flexible date range filtering
  - Multi-domain processing

- **User-Friendly Features**:
  - Excel file upload (supports up to 1GB)
  - Custom date range selection
  - Multiple solution domain processing
  - Separate duplicate entries analysis view
  - Download results in Excel or CSV format

## Models Used 🤖

### Semantic Similarity Analysis
- **Model**: all-MiniLM-L6-v2 (SentenceTransformer)
- **Architecture**: BERT-based model optimized for semantic similarity
- **Features**: 
  - 384-dimensional embeddings
  - Cosine similarity comparison
  - 0.95 threshold for duplicate detection
  - Cross-domain duplicate detection support

### Sentiment and Importance Classification
- **Model**: facebook/bart-large-mnli
- **Architecture**: BART-based zero-shot classification
- **Features**:
  - Multi-label classification
  - Custom sentiment categories: highRating+, highRating, Neutral, Negative, Negative-
  - Zero-shot learning capabilities

## Installation 🚀

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Requirements 📝

- Python 3.7+
- Required packages:
  - streamlit
  - pandas
  - numpy
  - sentence-transformers
  - transformers
  - torch
  - openpyxl

## Usage 💡

1. Start the application:
```bash
streamlit run app.py
```

2. Using the Interface:
   - Upload your Excel file through the sidebar
   - Select the desired sheet from your Excel file
   - Choose a specific domain or "ALL Domains"
   - Select time period (entire or custom range)
   - Click "Process Domain" to start analysis
   - View results and duplicate analysis
   - Download results in Excel or CSV format

3. To stop the server:
```bash
pkill -f streamlit
```

## Data Processing Details 📊

The analyzer performs several steps:
1. Validates and processes date formats (M/D/YYYY)
2. Combines 'Reason' and 'Additional Details' fields
3. Generates text embeddings for similarity analysis
4. Detects possible duplicates (both within and across domains)
5. Analyzes request importance and sentiment using the following categories:
   - **Request Importance Categories**:
     - `highRating+`: Highest priority requests
     - `highRating`: High priority requests
   - **Sentiment Categories**:
     - `Neutral`: Neutral or balanced sentiment
     - `Negative`: Negative sentiment
     - `Negative-`: Strongly negative sentiment
6. Presents results in an easy-to-read format with separate duplicate analysis

## Technical Features 🛠️

- **Text Processing**:
  - Advanced text cleaning and normalization
  - Batch processing for efficient handling of large datasets
  - Semantic similarity computation
  - Cross-domain duplicate detection
  
- **Performance**:
  - Efficient batch processing
  - Progress tracking for long operations
  - Memory-efficient handling of large files
  - GPU acceleration when available

## Output Format 📋

The processed data includes:
- Original row numbers for reference
- Solution domain classification
- Created date (M/D/YYYY format)
- Combined reason and additional details
- Possible duplicates identification (within and across domains)
- Request feature importance (highRating+, highRating)
- Sentiment analysis (Neutral, Negative, Negative-)

## Notes 📌

- Date format displayed as M/D/YYYY for consistency
- Duplicate detection uses a 0.95 similarity threshold
- Processing time varies based on data size and selected domains
- The application uses CPU by default but will automatically use CUDA if available
- Cross-domain duplicates are tracked separately from same-domain duplicates

## Support 🤝

For issues, questions, or contributions, please [create an issue](your-issue-tracker-url) or contact [your-contact-info].

## License 📄

© 2024 Solutions Domain Analyzer. All rights reserved. 