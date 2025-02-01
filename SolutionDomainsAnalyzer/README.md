# Solutions Domain Analyzer 🔍

A powerful Streamlit application for analyzing and processing solutions domain data with advanced NLP capabilities. The application supports both OpenAI and BridgeIT authentication methods.

## Models Used

### Local Models (No External API Calls)
- **Sentiment Analysis**: facebook/bart-large-mnli (Zero-shot classifier)
  - Runs locally for sentiment and feature importance classification
  - No data sent to external services
  
- **Text Embeddings**: all-MiniLM-L6-v2 (SentenceTransformer)
  - Local embedding generation for duplicate detection
  - Fully offline processing
  - 384-dimensional embeddings

### Cloud-Based Components
- **Fuzzy Search**: Langchain with BridgeIT Authentication
  - Secure enterprise authentication
  - Internal Cisco infrastructure
  - No data sent to public cloud services

## Features

### Data Processing
- Excel file upload support (up to 1GB)
- Multiple sheet handling
- Automatic date format handling
- Support for ALL domains or individual domain analysis

### Analysis Capabilities
- Sentiment Analysis
- Feature Request Importance Rating
- Duplicate Detection (within and across domains)
- Cross-Domain Analysis
- Text Similarity Matching

### Search Functionality
- Advanced Fuzzy Search using LLM
- Customer Name Search
- Date Range Filtering
- Intelligent Query Processing
- Case-insensitive Matching

### Export Options
- Excel Export
- CSV Export
- Filtered Results Export
- Duplicate Analysis Export

## Setup Instructions

### Prerequisites
```bash
python 3.12+
pip
virtualenv (recommended)
```

### Environment Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
# For BridgeIT Authentication
app_key=your_app_key
client_id=your_client_id
client_secret=your_client_secret
langsmith_api_key=your_langsmith_api_key

# For OpenAI Authentication
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application
```bash
streamlit run app.py
```

## Usage Guide

### Data Upload
1. Use the sidebar to upload your Excel file
2. Select the appropriate sheet from the dropdown
3. Choose between "ALL Domains" or a specific domain
4. Select time period (Entire or Custom)
5. Click "Process Domain"

### Search Capabilities
The application supports various search queries:
- Sentiment-based: "Show me ALL rows that have exactly Negative sentiment"
- Date-based: "Show me records between March 15th 2024 and March 16th 2024"
- Account-based: "Show me rows where account name contains at&t"
- Rating-based: "Show me all rows that have exactly highRating"
- Partial matches: "Show me rows where Solution Domain contains campus"
- Combined queries: "Show me rows where account name contains at&t and sentiment is exactly Negative"

Note: Use the word "exactly" in your query for exact matches, otherwise the search will use case-insensitive partial matching.

### Data Analysis
- Main Analysis tab shows complete processed data
- Fuzzy Search tab enables advanced search capabilities
- Duplicate analysis with cross-domain detection
- Customer search functionality
- Export capabilities for all views

## Technical Details

### Authentication Methods
The application supports two authentication methods:
1. BridgeIT Authentication (using OAuth2)
2. OpenAI Authentication (using API key)

### Search Implementation
- Two-step search process using LLM
- Row number indexing for accurate results
- Case-insensitive matching
- Pandas DataFrame operations
- Retry mechanism for reliability

### Data Processing
- Sentiment analysis using zero-shot classification
- Text embedding using SentenceTransformer
- Cosine similarity for duplicate detection
- Robust error handling and logging

## NGINX Proxy Configuration

### Creating NGINX Proxy and SSL Certificate

To enable HTTPS on your Streamlit server using a self-signed certificate, follow these steps:

### Step 1: Generate Self-Signed SSL Certificate

1. Create a directory for certificates:
```bash
mkdir sslcert
```

2. Generate private key and certificate:
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```

### Step 2: Install and Configure NGINX

1. Install NGINX:
```bash
sudo apt update
sudo apt install nginx
```

2. Configure NGINX as reverse proxy:
```bash
sudo vi /etc/nginx/sites-available/default
```

3. Add the following configuration:
```nginx
server {
    listen 443 ssl;
    server_name your_domain_or_ip;

    ssl_certificate /home/ubuntu/sslcert/mycert.pem;
    ssl_certificate_key /home/ubuntu/sslcert/mykey.key;

    client_max_body_size 200M;  # allows file upload up to 200MB

    location / {
        proxy_pass http://localhost:8501;  # Default Streamlit port
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

4. Test and reload NGINX:
```bash
sudo nginx -t
sudo systemctl reload nginx
```

Your Streamlit application will now be accessible via HTTPS with SSL encryption.

## Troubleshooting

### Common Issues
1. Authentication Errors:
   - Verify environment variables are correctly set
   - Check token expiration and refresh process

2. Search Issues:
   - Ensure query format is clear and specific
   - Check for case sensitivity in search terms
   - Verify data is properly processed before searching

3. Performance Issues:
   - Large files may require additional processing time
   - Consider filtering data before processing
   - Check system resources and memory usage

## Support

For issues and feature requests, please contact the development team or create an issue in the repository.

## License

© 2024 Solutions Domain Analyzer. All rights reserved. 