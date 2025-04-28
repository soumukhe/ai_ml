# Cisco Documentation RAG System

A Retrieval-Augmented Generation (RAG) system built specifically for Cisco documentation, using BridgeIT's GPT-4 API and FAISS vector store for efficient document retrieval and question answering.

## Features

- **Multi-User Support**: Individual workspaces for different users with separate document storage and chat histories
- **Document Management**: Upload and process PDF documents with automatic text extraction and chunking
- **Semantic Search**: Uses FAISS vector store with HuggingFace embeddings for efficient document retrieval
- **Intelligent Q&A**: Leverages Cisco's BridgeIT GPT-4 API for accurate and contextual responses
- **Chat History**: Maintains conversation history for better context awareness, with option to clear per user
- **Favorites System**: Save and manage favorite Q&A pairs
- **User Profiles**: Customizable user profiles with tags and descriptions
- **Document Sharing**: Ability to copy documents between users
- **UI Customization**: Toggle between light and dark themes for comfortable viewing

## Prerequisites

- Python 3.8+
- Cisco BridgeIT API credentials (app_key, client_id, client_secret)
- Conda (optional, if using Conda environment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cisco-rag
```

2. Create and activate a virtual environment:

   Option 1 - Using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Option 2 - Using Conda:
   ```bash
   conda create -n cisco_rag python=3.12  # or your preferred Python version
   conda activate cisco_rag
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```env
app_key = 'your-app-key'
client_id = 'your-client-id'
client_secret = 'your-client-secret'
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Features available in the UI:
   - Upload PDF documents
   - Ask questions about the documents
   - View and manage chat history
   - Clear chat history per user
   - Toggle between light and dark themes
   - Save favorite Q&A pairs
   - Switch between users
   - Manage user profiles
   - Share documents between users

## Architecture

- **Frontend**: Streamlit web interface with theme customization
- **Document Processing**: PDFMiner for text extraction, RecursiveCharacterTextSplitter for chunking
- **Vector Store**: FAISS with HuggingFace embeddings (all-MiniLM-L6-v2)
- **LLM Integration**: Custom BridgeITLLM class using Cisco's GPT-4 API
- **Storage**: File-based storage for documents, embeddings, and user data

## File Structure

```
.
├── app.py                 # Streamlit web application
├── rag_system.py         # Core RAG system implementation
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
└── user_data/           # User-specific data storage
    └── {user_id}/
        ├── docs/       # PDF documents
        ├── db/         # FAISS vector store
        ├── metadata.json
        ├── history.json
        ├── favorites.json
        └── profile.json
```

## Security

- OAuth2 authentication with Cisco BridgeIT API
- Secure token management with automatic refresh
- User data isolation
- Environment variable based credential management
- **Local Document Processing**:
  - All document embeddings are generated locally using HuggingFace's all-MiniLM-L6-v2 model
  - No document content is sent to external servers for embedding
  - The embedding model runs entirely on your local CPU
  - Only the final questions are sent to Cisco's BridgeIT API
  - All embeddings and vector stores are saved locally in the user's workspace

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add appropriate license information]

## Acknowledgments

- Cisco BridgeIT team for API access
- LangChain for the foundation components
- FAISS for vector storage
- HuggingFace for embeddings (local model)
