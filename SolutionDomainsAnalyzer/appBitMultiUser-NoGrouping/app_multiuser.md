# Solutions Domain Analyzer - Multi-User Documentation

## Overview
The Solutions Domain Analyzer is a powerful web application built with Streamlit that enables users to analyze solution domain data with advanced NLP capabilities. It supports multiple users, secure authentication, and provides comprehensive analysis tools including sentiment analysis, duplicate detection, and theme analysis.

## Architecture

### Component Structure
```
SolutionDomainsAnalyzer/
├── app_multiuser_BIT.py     # Main application file
├── login.py                 # User authentication and session management
├── user_manager.py         # User directory and data management
├── file_manager.py         # File operations and hash tracking
├── requirements.txt        # Dependencies
└── user_data/             # User-specific data storage
    ├── user1/
    │   ├── uploads/       # User's uploaded files
    │   ├── temp/         # Temporary files
    │   ├── reports/      # Generated reports
    │   ├── metadata.json # User metadata
    │   └── file_hashes.json
    └── user2/
        └── ...
```

### Key Components
1. **User Management System**
   - Secure user authentication
   - User-specific data isolation
   - Session state management
   - File access control

2. **File Management System**
   - Secure file uploads
   - Hash-based file tracking
   - Duplicate detection
   - Automatic cleanup

3. **Analysis Engine**
   - Sentiment analysis
   - Duplicate detection
   - Theme analysis
   - Fuzzy search capabilities

## Models and Technologies

### LLM (Language Model)
- **Model**: Azure OpenAI GPT-4
- **Usage**: Theme analysis, fuzzy search, and natural language processing
- **Authentication**: Uses Cisco's BridgeIT secure authentication

### Embedding Model
- **Model**: SentenceTransformers (all-MiniLM-L6-v2)
- **Dimensions**: 384
- **Usage**: Document similarity, duplicate detection
- **Performance**: Optimized for both CPU and GPU

### Sentiment Analysis
- **Model**: facebook/bart-large-mnli
- **Type**: Zero-shot classifier
- **Labels**: ["highRating+", "highRating", "Neutral", "Negative", "Negative-"]

### Semantic Search
- Implementation: Custom semantic search using cosine similarity
- Threshold: 0.95 for duplicate detection
- Batch Processing: Optimized for both GPU (128) and CPU (32)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Setup Steps

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd SolutionDomainsAnalyzer
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file with the following variables:
   ```
   app_key=<your-app-key>
   client_id=<your-client-id>
   client_secret=<your-client-secret>
   ```

5. **Create required directories**
   ```bash
   mkdir -p user_data reports
   chmod 775 user_data reports
   ```

### Running the Application
```bash
streamlit run app_multiuser_BIT.py
```

## Functionality

### 1. User Management
- User registration and login
- Secure session management
- User-specific data isolation
- Automatic directory creation

### 2. File Management
- Secure file uploads
- Hash-based file tracking
- Automatic duplicate detection
- File cleanup and organization

### 3. Main Analysis
- Solution domain analysis
- Sentiment classification
- Duplicate detection
- Customer search functionality
- Export capabilities (Excel, CSV)

### 4. Fuzzy Search
- Natural language queries
- Advanced filtering options
- Cross-domain search
- Export search results

### 5. Reports
- Theme analysis
- PDF report generation
- Markdown export
- Statistical summaries

### 6. Data Processing
- Automatic text cleaning
- Batch processing
- GPU acceleration when available
- Progress tracking

## Security Features

1. **User Isolation**
   - Separate directories for each user
   - Secure file access control
   - Session state management

2. **File Security**
   - Hash verification
   - Secure file handling
   - Automatic cleanup

3. **API Security**
   - Secure token management
   - Environment variable protection
   - Rate limiting

## Best Practices

1. **File Management**
   - Regular cleanup of temporary files
   - Monitoring storage usage
   - Backing up important data

2. **Performance**
   - Batch processing for large datasets
   - GPU acceleration when available
   - Caching for improved response times

3. **Error Handling**
   - Comprehensive error logging
   - User-friendly error messages
   - Automatic recovery mechanisms

## Troubleshooting

### Common Issues
1. **File Upload Issues**
   - Check file permissions
   - Verify file format
   - Check storage space

2. **Processing Errors**
   - Verify input data format
   - Check system resources
   - Review error logs

3. **Authentication Issues**
   - Verify environment variables
   - Check network connectivity
   - Review authentication logs

## Support and Maintenance

### Logging
- Comprehensive error logging
- Performance monitoring
- User activity tracking

### Updates
- Regular dependency updates
- Security patches
- Feature enhancements

### Backup
- Regular backup of user data
- Configuration backup
- System state preservation

## Future Enhancements
1. Enhanced visualization capabilities
2. Advanced reporting features
3. Integration with additional data sources
4. Performance optimizations
5. Extended API functionality

## Contributing
Please follow these guidelines when contributing:
1. Follow PEP 8 style guide
2. Write comprehensive tests
3. Document all changes
4. Create detailed pull requests

## License
[Specify your license information here] 