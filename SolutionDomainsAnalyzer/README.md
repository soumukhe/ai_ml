# Solutions Domain Analyzer 🔍

A powerful web application that analyzes solutions domain data using advanced Natural Language Processing (NLP) techniques. This tool helps identify duplicate entries, assess request importance, and analyze sentiment in your solutions domain data.

## Features ✨

- **Interactive Web Interface**: Built with Streamlit for a clean, professional user experience
- **Advanced NLP Processing**:
  - Text embedding generation using SentenceTransformer
  - Semantic similarity detection
  - Zero-shot classification for request importance and sentiment analysis
- **Data Processing**:
  - Automatic duplicate detection
  - Request importance classification (highlyRequested+, highlyRequested)
  - Sentiment analysis (Neutral, Negative, Negative-)
- **User-Friendly Features**:
  - Progress tracking with detailed status updates
  - Excel file upload (supports up to 1GB)
  - Multiple solution domain processing
  - Download results in Excel or CSV format
  - Highlighted visualization of results

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

## Usage 💡

1. Start the application:
```bash
./run_app.sh
```
The application will automatically open in your default web browser.

2. Using the Interface:
   - Upload your Excel file using the sidebar
   - Select the desired sheet from your Excel file
   - Choose a Solution Domain to analyze
   - Click "Process Domain" to start the analysis
   - Download results in Excel or CSV format

3. To stop the server:
```bash
pkill -f streamlit
```

## Data Processing Details 📊

The analyzer performs several steps:
1. Combines 'Reason' and 'Additional Details' fields
2. Generates text embeddings for similarity analysis
3. Detects possible duplicates using cosine similarity
4. Analyzes request importance and sentiment
5. Presents results in an easy-to-read format

## Technical Features 🛠️

- **Text Processing**:
  - Advanced text cleaning and normalization
  - Batch processing for efficient handling of large datasets
  - Semantic similarity computation
  
- **Machine Learning Models**:
  - SentenceTransformer ('all-MiniLM-L6-v2') for embeddings
  - BART-large-MNLI for zero-shot classification
  - Configurable similarity thresholds

- **Performance**:
  - Efficient batch processing
  - Progress tracking for long operations
  - Memory-efficient handling of large files

## Output Format 📋

The processed data includes:
- Original data fields
- Combined Reason with Additional Details
- Possible duplicates identification
- Request importance classification
- Sentiment analysis results

## Requirements 📝

- Python 3.8+
- See requirements.txt for complete package list

## Notes 📌

- The application uses CPU by default but will automatically use CUDA if available
- Processing time depends on the size of your dataset and available computational resources
- Duplicate detection threshold is set to 0.95 (configurable)

## License 📄

[Your License Information]

## Support 🤝

For issues, questions, or contributions, please [create an issue](your-issue-tracker-url) or contact [your-contact-info].

---
© 2024 Solutions Domain Analyzer 