# PDF Document Question Answering System

A Python project that demonstrates how to build an Agentic RAG (Retrieval Augmented Generation) system for PDF documents using Streamlit, OpenAI, PydanticAI, and Supabase. The system leverages intelligent agents to provide enhanced document retrieval and question answering capabilities through a modern web interface.

## Overview

This project implements an advanced document question-answering system that:
- Uses PydanticAI agents for intelligent document retrieval and reasoning
- Processes PDF documents and stores their content in Supabase with vector embeddings
- Provides a user-friendly Streamlit interface for document management and querying
- Implements multiple agent tools for comprehensive document analysis
- Maintains chat history and allows for document management

## Project Structure

- `streamlit_app.py`: Main Streamlit application with the user interface
- `crawl_pdf_docs.py`: Handles PDF processing, chunking, embedding generation, and Supabase storage
- `rag_agentic.py`: Implements the agentic RAG system with multiple specialized tools using PydanticAI
- `run_streamlit.sh`: Script to run the Streamlit application in the background
- `requirements.txt`: Project dependencies
- `pdf_pages.sql`: SQL schema for Supabase database setup

## Features

### Agentic Question Answering
- Multiple specialized agent tools for comprehensive document analysis:
  1. Document retrieval tool for finding relevant content
  2. Document listing tool for content overview
  3. Page content tool for detailed analysis
- Context-aware responses using intelligent agents
- Real-time chat interface with agent-driven responses

### Document Management
- Upload PDF documents (up to 1GB per file)
- Process documents with real-time progress tracking
- Delete documents with automatic cleanup in both filesystem and database
- Database cleanup tool for orphaned records

### Chat History
- Persistent chat history
- Collapsible Q&A pairs
- Individual and bulk deletion options

## Setup

### 1. Clone the Repository

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
echo "PydanticAI_Agents/pydanticAgenticRagSuperbase" >> .git/info/sparse-checkout

# Pull the subdirectory
git pull origin master

# Enter the project directory
cd PydanticAI_Agents/pydanticAgenticRagSuperbase
```

### 2. Environment Setup

Create a virtual environment:
```bash
# Using conda
conda create -n pdf_qa python=3.12
conda activate pdf_qa

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini  # or your preferred model
```

### 5. Supabase Setup

1. Create the database schema:
   - Execute the contents of `pdf_pages.sql` in your Supabase instance
   - This creates the `pdf_pages` table with vector support and necessary indexes

2. Verify your Supabase connection before proceeding

## Usage

### 1. Start the Application

Run the Streamlit application:
```bash
./run_streamlit.sh
```

The UI will be available at: http://localhost:8501

### 2. Using the Interface

1. Document Management Tab:
   - Upload PDF documents using the file uploader
   - Click "Process Documents" to extract and embed content
   - Use the delete buttons to remove documents
   - Use "Clean DB" to remove orphaned records

2. Chat Tab:
   - Enter questions about your documents
   - View responses with context from the PDFs
   - Continue the conversation naturally

3. History Tab:
   - View all previous Q&A pairs
   - Expand/collapse answers
   - Delete individual conversations or all history

## Technical Details

### PDF Processing Pipeline

1. Document Upload:
   - Files are saved to the `pdf_docs` directory
   - Size limit of 1GB per file

2. Processing:
   - Text extraction from PDFs
   - Content chunking with smart breaks
   - Embedding generation using OpenAI
   - Storage in Supabase with metadata

3. Querying:
   - Vector similarity search
   - Context aggregation
   - Response generation using PydanticAI agents

### Database Schema

The `pdf_pages` table stores:
- Document content chunks
- Vector embeddings
- Metadata (file name, page numbers, timestamps)
- Titles and summaries

## Troubleshooting

### Common Issues

1. Streamlit Access:
   - If the UI doesn't open automatically, manually visit http://localhost:8501
   - To stop the application: `pkill -f streamlit`

2. Document Processing:
   - Large documents may take longer to process
   - Progress bar and status updates show processing status
   - Check the terminal for detailed logs

3. Database Cleanup:
   - Use the "Clean DB" button if documents show in UI but queries don't work
   - This removes orphaned records from deleted documents

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 