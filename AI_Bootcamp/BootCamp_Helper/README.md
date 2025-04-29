# AI Knowledge Assistant - CX Delivered AI/ML Bootcamp

![AI Assistant](https://img.shields.io/badge/AI-Assistant-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-green)
![LangGraph](https://img.shields.io/badge/LangGraph-Workflow-orange)

A sophisticated AI-powered knowledge assistant built using Streamlit, LangGraph, and advanced language models. This application provides intelligent responses to queries about AI and machine learning concepts from the CX Delivered AI/ML bootcamp material.

## 🚀 Features

- Interactive Streamlit-based user interface
- Advanced RAG (Retrieval Augmented Generation) system
- LangGraph-based workflow for intelligent response generation
- Vector-based semantic search using ChromaDB
- Azure OpenAI integration for advanced language processing
- GPU acceleration support (CUDA/Metal)

## 🛠️ Architecture

### Components

1. **Vector Store (ChromaDB)**
   - Uses `sentence-transformers/all-mpnet-base-v2` for embeddings
     - 768-dimensional embeddings
     - Optimized for semantic similarity tasks
     - Supports GPU acceleration for faster processing
   - Text Chunking Strategy:
     - Uses `SemanticChunker` from `langchain_experimental`
     - Splits text based on semantic boundaries rather than fixed sizes
     - Preserves context and meaning within chunks
     - Configurable chunk size and overlap
   - Data Storage:
     - Document chunks from PowerPoint files
     - Hypothetical questions generated for each chunk
     - Metadata including source information and relationships
     - Collection name: "hypothetical_questions"
   - Persists data in `./rag_db` directory
   - Supports GPU acceleration for faster processing
   - Custom `EmbeddingWrapper` class for efficient batch processing
   - Batch size optimization (32 for CPU, 128 for GPU)

2. **Language Models**
   - Azure OpenAI GPT-4o-mini for response generation
   - Sentence Transformers for text embeddings
   - Custom token refresh mechanism for Azure authentication

3. **LangGraph Workflow**
   The complete workflow can be visualized at [Memraid Live](https://memraid.live) using the following configuration:

   ```mermaid
   ---
   config:
     flowchart:
       curve: linear
   ---
   graph TD;
       __start__([<p>__start__</p>]):::first
       modify_query(modify_query)
       retrieve_hypothetical_questions(retrieve_hypothetical_questions)
       retrieve_context(retrieve_context)
       check_context_exists(check_context_exists)
       no_context_response(no_context_response)
       craft_response(craft_response)
       score_groundedness(score_groundedness)
       not_relevant_response(not_relevant_response)
       refine_response(refine_response)
       check_precision(check_precision)
       refine_query(refine_query)
       max_iterations_reached(max_iterations_reached)
       __end__([<p>__end__</p>]):::last
       __start__ --> modify_query;
       check_context_exists -. &nbsp;yes&nbsp; .-> craft_response;
       check_context_exists -. &nbsp;no&nbsp; .-> no_context_response;
       check_precision -.-> max_iterations_reached;
       check_precision -.-> refine_query;
       craft_response --> score_groundedness;
       modify_query --> retrieve_hypothetical_questions;
       refine_query --> retrieve_hypothetical_questions;
       refine_response --> craft_response;
       retrieve_context --> check_context_exists;
       retrieve_hypothetical_questions --> retrieve_context;
       score_groundedness -.-> check_precision;
       score_groundedness -.-> max_iterations_reached;
       score_groundedness -. &nbsp;not_grounded&nbsp; .-> not_relevant_response;
       score_groundedness -.-> refine_response;
       check_context_exists -.-> __end__;
       max_iterations_reached --> __end__;
       no_context_response --> __end__;
       not_relevant_response --> __end__;
       classDef default fill:#f2f0ff,line-height:1.2
       classDef first fill-opacity:0
       classDef last fill:#bfb6fc
   ```

   To generate the workflow visualization in your code:
   ```python
   WORKFLOW_APP = create_workflow().compile()
   # Get the Mermaid syntax representation
   mermaid_syntax = WORKFLOW_APP.get_graph().draw_mermaid()
   print(mermaid_syntax)
   ```

### Workflow Function Descriptions

1. **modify_query**
   - Enhances the user's query with additional context and hypothetical questions
   - Improves search relevance by expanding the query scope

2. **retrieve_hypothetical_questions**
   - Generates potential follow-up questions based on the original query
   - Helps in retrieving more comprehensive context

3. **retrieve_context**
   - Searches the vector store for relevant documents
   - Uses semantic similarity to find the most relevant content

4. **check_context_exists**
   - Validates if sufficient context was found
   - Routes the workflow based on context availability

5. **no_context_response**
   - Handles cases where no relevant context is found
   - Provides a graceful response to the user

6. **craft_response**
   - Generates a comprehensive answer using the retrieved context
   - Ensures the response is well-structured and informative

7. **score_groundedness**
   - Evaluates how well the response is supported by the context
   - Determines if the response needs refinement

8. **not_relevant_response**
   - Handles cases where the response is not sufficiently grounded
   - Provides an alternative response strategy

9. **refine_response**
   - Improves the response based on groundedness feedback
   - Enhances accuracy and relevance

10. **check_precision**
    - Evaluates the precision of the response
    - Determines if further refinement is needed

11. **refine_query**
    - Modifies the search query based on precision feedback
    - Improves context retrieval

12. **max_iterations_reached**
    - Handles cases where maximum refinement attempts are reached
    - Provides the best available response

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with the following variables:
   ```
   app_key=your_app_key
   client_id=your_client_id
   client_secret=your_client_secret
   LANGSMITH_API_KEY=your_langsmith_key
   ```

## 🏃‍♂️ Usage

### Database Setup (One-time)
Run the database creation script to process PowerPoint files:
```bash
python createdb.py
```

### Running the Application
Start the Streamlit application:
```bash
./run_app.sh
```
The application will be available at `http://localhost:8501`

## 🔧 Key Functions

### Database Creation (`createdb.py`)
- `check_gpu_availability()`: Detects and configures GPU support
- `init_azure_openai()`: Initializes Azure OpenAI with token refresh
- `EmbeddingWrapper`: Custom wrapper for sentence transformer embeddings
- `embed_function()`: Handles batch processing of text embeddings

### Application (`app.py`)
- `initialize_components()`: Sets up LLM, vector store, and retriever
- `create_chain()`: Creates RAG chain with custom prompt template
- `retrieve_hypothetical_questions()`: Retrieves relevant questions
- `craft_response()`: Generates AI responses
- `score_groundedness()`: Evaluates response quality
- `check_precision()`: Ensures response accuracy

## 📚 Dependencies

Key packages used:
- `streamlit`: Web interface
- `langchain-core`: Core LangChain functionality
- `langchain-community`: Community extensions
- `sentence-transformers`: Text embeddings
- `chromadb`: Vector database
- `torch`: Deep learning framework
- `langgraph`: Workflow management

## 🔒 Security

- Secure token management for Azure OpenAI
- Environment variable-based configuration
- No hardcoded credentials

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CX Delivered AI/ML Bootcamp
- OpenAI and Azure OpenAI
- LangChain and LangGraph communities 