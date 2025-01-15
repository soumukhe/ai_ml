# Document Q&A System

A powerful document question-answering system that combines web crawling, PDF processing, and AI-powered search to help you interact with your documentation.

## Features

### Document Sources
1. **Web Crawling**
   - Parallel crawler with configurable concurrency (up to 10 processes)
   - Smart rate limiting with adjustable delays
   - Support for both sitemap URLs and direct webpage URLs
   - JavaScript-enabled crawling for dynamic content
   - Automatic HTML to Markdown conversion

2. **PDF Documents**
   - Direct PDF upload through the UI
   - Support for files up to 1GB
   - Automatic PDF to Markdown conversion
   - Structure preservation (headings, lists, paragraphs)
   - Batch processing capabilities

### Processing Pipeline
1. **Document Ingestion**
   - Smart document structure detection
   - Format preservation
   - Automatic metadata extraction
   - Change detection via MD5 hashing
   - Persistent JSON storage of document hashes
   - Automatic detection of modified files

2. **Text Processing**
   - Configurable chunk size (500-3000 tokens)
   - Smart chunking that preserves document structure
   - Metadata tracking for source attribution
   - Improved code block preservation with larger chunks (2500 tokens)
   - Structured response format with dedicated sections

3. **Vector Storage**
   - OpenAI's `text-embedding-3-small` model (1536 dimensions)
   - ChromaDB with PersistentClient for reliable storage
   - JSON-based state management for embeddings and hashes
   - Efficient document reloading without re-embedding
   - Automatic verification of document changes

4. **Query Processing**
   - Semantic similarity search
   - Multi-document context assembly
   - Source attribution
   - Conversation history management

### User Interface
- Clean, modern Streamlit interface
- Dark/Light mode toggle
- Document management dashboard with advanced features:
  - Search and filtering by content type
  - Multiple sorting options
  - Detailed document statistics and analysis
  - Batch operations (export, delete)
- Chat interface with markdown support
- Persistent chat history with conversation management:
  - View and delete individual conversations
  - Export chat history
  - Automatic saving across sessions

## System Requirements

### Python Version
- Compatible with Python 3.8 and above, including Python 3.12
- Uses standard libraries and well-maintained packages
- No version-specific dependencies

### Operating System Dependencies

#### macOS
```bash
brew install tesseract poppler opencv
pip install python-magic  # macOS-specific dependency
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils libmagic1 python3-opencv
pip install python-magic  # Linux-specific dependency
```

#### Windows
1. Install Tesseract OCR:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add installation directory to system PATH
2. Install Poppler:
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Add installation directory to system PATH
3. Windows-specific dependencies:
   ```bash
   pip install python-magic-bin opencv-python
   ```

### Installation Steps

1. Clone the Repository

   To clone only this specific project (sparse checkout):
   ```bash
   # Create and enter a new directory
   mkdir my_demo && cd my_demo

   # Initialize git
   git init

   # Add the remote repository
   git remote add -f origin https://github.com/soumukhe/ai_ml.git

   # Enable sparse checkout
   git config core.sparseCheckout true

   # Specify the subdirectory you want to clone
   echo "RAG_With_Crawl4AI_And_PDF" >> .git/info/sparse-checkout

   # Pull the subdirectory
   git pull origin master

   # Enter the project directory
   cd RAG_With_Crawl4AI_And_PDF
   ```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Upgrade pip first
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Install browser for web crawling
playwright install
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
ALLOW_RESET=true  # Enable ChromaDB reset functionality
```

## Usage

1. Start the application (choose one method):

   **Method 1**: Using the shell script (recommended for Unix/Linux/macOS):
   ```bash
   ./run_streamlit.sh
   # To stop: pkill -f streamlit
   ```

   **Method 2**: Direct Streamlit command:
   ```bash
   streamlit run streamlit_app.py
   ```

2. Access the web interface at `http://localhost:8501`

### Adding Documents

#### Web Crawling
1. Go to the Document Management tab
2. Enter a sitemap URL or webpage URL
3. Configure crawling parameters (optional)
4. Click "Start Crawling"

#### PDF Upload
1. Go to the Document Management tab
2. Use the PDF upload section
3. Select your PDF file (up to 1GB)
4. Wait for conversion to complete
5. Click "Reprocess Documents" if needed

### Asking Questions
1. Navigate to the Chat tab
2. Type your question
3. Press Enter or click "Ask"
4. View AI-generated response with source references

### Managing Documents
- Use the Document Management tab to:
  - View all documents with detailed statistics
  - Search by content and file names
  - Filter by content type (code, links, headings)
  - Sort by name, size, modification date, or content metrics
  - View detailed file analysis
  - Perform batch operations (export, delete)
  - Import/Export markdown files
  - Process PDF documents up to 1GB

### Chat History
- Access the Chat History tab to:
  - View past conversations grouped by Q&A pairs
  - Expand/collapse individual conversations
  - Delete specific conversations
  - Clear entire chat history
  - Conversations automatically persist across sessions
  - JSON-based storage for reliability

## Troubleshooting

### Common Issues

1. PDF Processing
   - Ensure PDF files are not corrupted or password-protected
   - Verify file size is under 1GB
   - Check file permissions

2. ChromaDB Issues
   - For persistence issues, try:
     ```bash
     rm -rf ./chroma_db/*  # Clear ChromaDB directory
     ```
   - Ensure ALLOW_RESET=true in .env file
   - Check file permissions on chroma_db directory
   - Verify sufficient disk space

3. Web Crawling Issues
   - Verify URL accessibility
   - Check JavaScript requirements
   - Adjust concurrent processes (1-10) and delay (0-5s) if needed
   - Monitor console output for crawling progress

### Error Messages

If you encounter errors:
1. Check the console output for detailed error messages
2. Verify all dependencies are correctly installed
3. Ensure system requirements are met
4. Check file permissions in the data and chroma_db directories

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your chosen license]

### Persistent Storage
The system now uses a robust persistent storage system:
1. **Document Hashes**: Stored in `chroma_db/document_hashes.json`
   - Tracks MD5 hashes of all processed files
   - Used for change detection on restart
   - Prevents unnecessary reprocessing

2. **Document Embeddings**: Stored in `chroma_db/documents.json`
   - Contains all document chunks and their embeddings
   - Loaded automatically on system restart
   - Enables quick system recovery

3. **Chat History**: Stored in `chroma_db/chat_history.json`
   - Persists all conversations across sessions
   - Allows conversation management and deletion
   - Automatically saved after each interaction

### Response Format
Responses now follow a structured format:
1. **Overview**: High-level summary of the topic
2. **Components and Technologies**: Key technical elements
3. **How It Works**: Detailed explanation of processes
4. **Technical Details**: Specific implementation details
5. **Code Example**: Relevant code snippets (when applicable)
6. **Additional Information**: Supplementary context