# Documentation RAG System

A Retrieval-Augmented Generation (RAG) system built for documentation, using OpenAI's GPT-4o-mini model and FAISS vector store for efficient document retrieval and question answering.

## Features

- **Multi-User Support**: Individual workspaces for different users with separate document storage and chat histories
- **Document Management**: Upload and process PDF documents with automatic text extraction and chunking
- **Semantic Search**: Uses FAISS vector store with HuggingFace embeddings for efficient document retrieval
- **Intelligent Q&A**: Leverages OpenAI's GPT-4o-mini model for accurate and contextual responses
- **Chat History**: Maintains conversation history for better context awareness, with option to clear per user
- **Favorites System**: Save and manage favorite Q&A pairs
- **User Profiles**: Customizable user profiles with tags and descriptions
- **Document Sharing**: Ability to copy documents between users
- **UI Customization**: Toggle between light and dark themes for comfortable viewing

## Prerequisites

- Python 3.8+
- OpenAI API Key (Required - Get one from [OpenAI Platform](https://platform.openai.com/api-keys))
- Conda (optional, if using Conda environment)

## Installation

1. Clone the specific project directory:
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
echo 'document-ragwGPT4-o-mini' >> .git/info/sparse-checkout

# Pull the subdirectory
git pull origin master

# Enter the project directory
cd document-ragwGPT4-o-mini
```

2. Create and activate a virtual environment:

   Option 1 - Using venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

   Option 2 - Using Conda:
   ```bash
   conda create -n doc_rag python=3.12  # or your preferred Python version
   conda activate doc_rag
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your OpenAI API key:
```env
OPENAI_API_KEY=your-api-key-here
```

⚠️ **Important**: The application requires a valid OpenAI API key to function. Make sure to:
1. Create an account on [OpenAI Platform](https://platform.openai.com) if you don't have one
2. Generate an API key in your OpenAI account settings
3. Copy the API key and paste it in the `.env` file as shown above
4. Keep your API key secure and never share it publicly

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
- **LLM Integration**: OpenAI GPT-4o-mini model via LangChain
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

- OpenAI API key management through environment variables
- User data isolation
- **Local Document Processing**:
  - All document embeddings are generated locally using HuggingFace's all-MiniLM-L6-v2 model
  - No document content is sent to external servers for embedding
  - The embedding model runs entirely on your local CPU
  - Only the final questions are sent to OpenAI's API for processing
  - All embeddings and vector stores are saved locally in the user's workspace
  - Questions and responses are processed through OpenAI's secure API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Add appropriate license information]

## Acknowledgments

- OpenAI for the GPT-4o-mini model
- LangChain for the foundation components
- FAISS for vector storage
- HuggingFace for embeddings (local model)
- Streamlit for the web interface

## Hosting on Ubuntu with NGINX Proxy

To host this application on Ubuntu with HTTPS support using a self-signed certificate and NGINX proxy:

### 1. Generate Self-Signed SSL Certificate

```bash
# Create directory for certificates
mkdir sslcert
cd sslcert

# Generate certificate and key
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```

When prompted, fill in the certificate information (country code, state, locality, etc.).

### 2. Install and Configure NGINX

```bash
# Install NGINX
sudo apt update
sudo apt install nginx
```

### 3. Configure NGINX as Reverse Proxy

Create or edit the NGINX configuration:

```bash
sudo vi /etc/nginx/sites-available/default
```

Add the following configuration:

```nginx
server {
    listen 443 ssl;
    server_name <serverPublicIP>;  # Replace with your server's public IP

    ssl_certificate /home/ubuntu/sslcert/mycert.pem;
    ssl_certificate_key /home/ubuntu/sslcert/mykey.key;

    client_max_body_size 200M;  # Allows file uploads up to 200MB

    location / {
        proxy_pass http://localhost:8501;  # Default Streamlit port
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### 4. Test and Reload NGINX

```bash
# Test the configuration
sudo nginx -t

# If test is successful, reload NGINX
sudo systemctl reload nginx
```

### 5. Run the Application

Start the Streamlit application as usual, and it will be accessible via HTTPS through the NGINX proxy.

**Note**: When using a self-signed certificate, browsers will show a security warning. This is normal for development/internal use. For production environments, consider using a proper SSL certificate from a trusted certificate authority.
