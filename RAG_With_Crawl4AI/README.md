# Document Q&A System with RAG and crawl4AI

A powerful document question-answering system that combines web crawling, RAG (Retrieval Augmented Generation), and a modern UI to provide intelligent answers from your documentation.

## Features

- 🤖 **Advanced RAG System**
  - Uses OpenAI's GPT-4o for intelligent responses
  - ChromaDB for efficient vector storage
  - Smart document chunking with code preservation
  - Automatic change detection and re-embedding

- 🌐 **Web Crawling with crawl4AI**
  - Automatic documentation crawling
  - Converts web pages to markdown format
  - Preserves code blocks and formatting
  - Links:
    - [GitHub Repository](https://github.com/unclecode/crawl4ai)
    - [Documentation](https://docs.crawl4ai.com)
    - [Example Agent](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent)
    - [Tutorial Video](https://www.youtube.com/watch?v=JWfNLF_g_V0)

- 💻 **Modern Streamlit UI**
  - Clean, professional interface
  - Real-time chat experience
  - Document status tracking
  - Conversation history management

## Technical Specifications

### Models and Database
- **Embedding Model**: OpenAI's `text-embedding-3-small`
  - Latest embedding model from OpenAI
  - Optimized for high-quality semantic search
  - 1536-dimensional embeddings
  - Excellent performance for technical documentation

- **Language Model**: OpenAI's `gpt-4o`
  - Advanced reasoning capabilities
  - High-quality code understanding
  - Context window: 128,000 tokens
  - Maximum response length: 16,384 tokens

- **Vector Database**: ChromaDB
  - Persistent storage for embeddings
  - Efficient similarity search
  - Automatic metadata handling
  - Real-time updates and changes detection
  - Collection-based organization for better scalability

### System Architecture
- **RAG Pipeline**:
  1. Document Processing → Chunking → Embedding
  2. Vector Storage in ChromaDB
  3. Similarity Search on Query
  4. Context Assembly and LLM Generation
  5. Response Formatting and Presentation

- **Performance Features**:
  - Smart document change detection
  - Efficient chunk size management (1500 tokens)
  - Code block preservation
  - Automatic metadata tracking
  - Conversation history management

## Installation

1. **Clone the Repository**

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
   echo "RAG_With_Crawl4AI" >> .git/info/sparse-checkout

   # Pull the subdirectory
   git pull origin master

   # Enter the project directory
   cd RAG_With_Crawl4AI
   ```

2. **Set Up Virtual Environment**

   Choose one of the following methods:

   **Using conda (recommended)**:
   ```bash
   # Create a new conda environment
   conda create -n crawl4ai_rag python=3.12
   
   # Activate the environment
   conda activate crawl4ai_rag
   ```

   **Using venv**:
   ```bash
   # Create a new virtual environment
   python -m venv venv
   
   # Activate the environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
```bash
# Install the required packages
pip install -r requirements.txt

# Install crawl4AI
pip install -U crawl4ai

# Run crawl4AI setup
crawl4ai-setup

# Install Playwright (required for crawl4AI)
python -m playwright install --with-deps chromium

# Verify crawl4AI installation
crawl4ai-doctor
```

4. **Environment Setup**
Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_key_here
```

## Usage

1. **Crawl Documentation**

   The system includes a parallel crawler that saves web pages as markdown files:
   ```bash
   # Run the parallel crawler
   python 3-crawl_parallel_saveMarkdown.py
   ```
   
   By default, it crawls `https://ai.pydantic.dev/sitemap.xml`. To crawl a different site:
   1. Open `3-crawl_parallel_saveMarkdown.py`
   2. Change the `sitemap_url` variable to your desired URL
   3. Run the script

   **Example URLs that work well:**
   - Documentation sites with sitemaps:
     - `https://docs.python.org/3/sitemap.xml`
     - `https://docs.streamlit.io/sitemap.xml`
     - `https://platform.openai.com/sitemap.xml`
   - Individual documentation pages:
     - `https://docs.python.org/3/library/index.html`
     - `https://docs.streamlit.io/library/api-reference`

   **Crawler Features:**
   - Parallel processing for faster crawling
   - Automatic rate limiting to respect server limits
   - Smart handling of JavaScript-rendered content
   - Preservation of code blocks and formatting
   - Automatic cleanup of HTML artifacts
   - Organized output in the `data` directory

   **Troubleshooting Tips:**
   - If the crawler seems slow: Adjust `max_concurrent` in the script
   - If getting rate limited: Increase `delay_between_requests`
   - For JavaScript-heavy sites: The crawler uses Playwright to render content
   - Memory issues: Reduce `max_concurrent` or split crawling into smaller batches
   - Missing content: Check if the site requires authentication or has robots.txt restrictions

   Alternatively, you can use crawl4AI directly:
   ```bash
   # Use crawl4AI to fetch documentation
   # Output will be saved in the data directory as markdown files
   crawl4ai <url> -o data/

   # For sites requiring JavaScript rendering
   crawl4ai <url> -o data/ --use-browser

   # To control crawling speed
   crawl4ai <url> -o data/ --delay 2
   ```

   Note: In Safari, to see source code, first enable "Show features for web developers" in Safari/Settings/Advanced tab.
   Then on any web page, right click and show source documents.

   **Best Practices:**
   1. Always check robots.txt before crawling
   2. Start with a small subset to test configuration
   3. Monitor the output quality in the `data` directory
   4. Adjust parameters based on the target site's characteristics
   5. Consider using site-specific sitemaps when available

2. **Start the Q&A System**
```bash
# Start the Streamlit UI
./run_streamlit.sh

# Access the UI at: http://localhost:8501
# To stop: pkill -f streamlit
```

## System Components

### RAG System (`main_rag.py`)
- Document embedding and retrieval
- Conversation management
- Integration with OpenAI's GPT-4o
- Smart document chunking and processing

### Web Interface (`streamlit_app.py`)
- Professional chat interface
- Document status monitoring
- Conversation history management
- Real-time responses

### Utility Scripts
- `run_streamlit.sh`: Background process management
- Automatic port configuration
- Helpful status messages

## Contributing

Feel free to submit issues and enhancement requests!

## License

[Add your license information here]