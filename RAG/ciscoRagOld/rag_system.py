import os
import shutil
import json
import base64
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import HumanMessage, SystemMessage
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import PyPDF2
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class BridgeITLLM(LLM):
    """Custom LLM class for BridgeIT."""
    
    app_key: str = ""
    client_id: str = ""
    client_secret: str = ""
    auth_url: str = "https://id.cisco.com"
    api_url: str = "https://chat-ai.cisco.com"
    token: Optional[str] = None
    token_expiry: Optional[float] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4000
    api_version: str = "2023-08-01-preview"
    client: Optional[AzureOpenAI] = None

    def __init__(self, **kwargs):
        # Get credentials from environment
        kwargs["app_key"] = os.getenv('APP_KEY', '')
        kwargs["client_id"] = os.getenv('CLIENT_ID', '')
        kwargs["client_secret"] = os.getenv('CLIENT_SECRET', '')
        
        # Initialize parent class with all required fields
        super().__init__(**kwargs)
        
        # Validate credentials
        if not all([self.app_key, self.client_id, self.client_secret]):
            missing = []
            if not self.app_key: missing.append("APP_KEY")
            if not self.client_id: missing.append("CLIENT_ID")
            if not self.client_secret: missing.append("CLIENT_SECRET")
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        # Initialize the token
        self._get_token()

    @property
    def _llm_type(self) -> str:
        return "bridgeit"

    def _get_token(self) -> None:
        """Get OAuth2 token."""
        try:
            url = f"{self.auth_url}/oauth2/default/v1/token"
            payload = "grant_type=client_credentials"
            
            # Create base64 encoded credentials exactly as in the working example
            value = base64.b64encode(
                f'{self.client_id}:{self.client_secret}'.encode('utf-8')
            ).decode('utf-8')
            
            headers = {
                "Accept": "*/*",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {value}",
                "User": f'{{"appkey": "{self.app_key}"}}'
            }
            
            response = requests.post(
                url,
                headers=headers,
                data=payload,
                timeout=30
            )
            
            response.raise_for_status()
            token_data = response.json()
            
            self.token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self.token_expiry = time.time() + expires_in
            
            # Initialize OpenAI client with new token
            self.client = AzureOpenAI(
                azure_endpoint=self.api_url,
                api_key=self.token,
                api_version=self.api_version
            )
            
        except Exception as e:
            print(f"\nError getting token: {str(e)}")
            print(f"Auth URL used: {self.auth_url}")
            raise

    def _call(
        self, 
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the chat completion call."""
        try:
            self._check_token()

            messages = [{"role": "user", "content": prompt}]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                user=f'{{"appkey": "{self.app_key}"}}'
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"\nError in chat completion: {str(e)}")
            print(f"API URL used: {self.api_url}")
            raise

    def _check_token(self) -> None:
        """Check if token needs to be refreshed."""
        if (
            not self.token or 
            not self.token_expiry or 
            time.time() + 300 >= self.token_expiry
        ):
            print("Token expired or about to expire. Refreshing...")
            self._get_token()

    def get_token_status(self) -> str:
        """Get current token status."""
        if not self.token or not self.token_expiry:
            return "No token available"
        
        time_remaining = self.token_expiry - time.time()
        if time_remaining <= 0:
            return "Token expired"
        
        minutes = int(time_remaining / 60)
        seconds = int(time_remaining % 60)
        return f"Token valid for {minutes}m {seconds}s"

class RAGSystem:
    def __init__(self, user_id: str = "default"):
        """Initialize RAG system for a specific user."""
        # Initialize utility methods first
        self._initialize_utility_methods()
        
        # Set up paths and directories
        self.user_id = user_id
        self.base_dir = "user_data"
        self.docs_dir = os.path.join(self.base_dir, user_id, "docs")
        self.db_dir = os.path.join(self.base_dir, user_id, "db")
        self.metadata_file = os.path.join(self.base_dir, user_id, "metadata.json")
        self.history_file = os.path.join(self.base_dir, user_id, "history.json")
        self.favorites_file = os.path.join(self.base_dir, user_id, "favorites.json")
        self.profile_file = os.path.join(self.base_dir, user_id, "profile.json")
        
        # Create user directories
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Load or initialize user data
        self.metadata = self._load_json(self.metadata_file, {})
        self.chat_history = self._load_json(self.history_file, [])
        self.favorites = self._load_json(self.favorites_file, [])
        self.profile = self._load_json(self.profile_file, {
            "description": "",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "tags": []
        })
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Initialize components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        try:
            self.llm = BridgeITLLM(
                temperature=0.2,  # Reduced for more focused responses
                max_tokens=8000   # Increased for longer responses
            )
        except Exception as e:
            print(f"Error initializing BridgeIT LLM: {str(e)}")
            print("Ensure app_key, client_id, and client_secret are set in .env file")
            raise
        
        # Initialize vectorstore and QA chain
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_vectorstore()

        # Define system prompt
        self.system_prompt = """You are a helpful AI assistant with access to a knowledge base of documents. 
        When answering questions:
        1. Use information from the provided documents
        2. If a term or concept was explained in a previous answer, you can refer to and build upon that explanation
        3. If you don't find specific information in the documents, say so clearly
        4. Always maintain context from previous questions in the conversation
        5. If a previous answer provides relevant context, use it to enhance your current answer
        
        Current conversation context: {chat_history}
        """

    def _initialize_utility_methods(self):
        """Initialize utility methods to ensure they exist before use."""
        def _load_json(self, file_path: str, default_value: any) -> any:
            """Load JSON file or return default value if file doesn't exist."""
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        return json.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
            return default_value

        def _save_json(self, file_path: str, data: any):
            """Save data to JSON file."""
            try:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Error saving {file_path}: {str(e)}")

        # Attach methods to the instance
        self._load_json = _load_json.__get__(self)
        self._save_json = _save_json.__get__(self)

    def _initialize_vectorstore(self):
        """Initialize vectorstore from saved data or create new."""
        try:
            self.vectorstore = FAISS.load_local(
                self.db_dir,
                self.embeddings,
                "docs"
            )
            print(f"Loaded existing vector database for user {self.user_id}")
            self._initialize_qa_chain()
        except Exception as e:
            print(f"No existing vector database found for user {self.user_id}: {str(e)}")

    def _initialize_qa_chain(self):
        """Initialize the QA chain with the vector store."""
        if self.vectorstore:
            # Create prompt template
            prompt = PromptTemplate(
                input_variables=["context", "chat_history", "question"],
                template="""You are a helpful AI assistant with access to a knowledge base of documents. Use the following guidelines to provide detailed, comprehensive answers:

                1. Thoroughly analyze all provided document context
                2. Include specific details, examples, and explanations from the documents
                3. If relevant, cite specific sections or quotes from the documents
                4. Organize your response in a clear, structured manner
                5. If a concept needs clarification, provide additional context
                6. If you find conflicting information, acknowledge and explain the differences
                7. If you're not certain about something, clearly state your level of confidence
                8. Use previous conversation context to enhance your current answer
                9. If the documents don't contain enough information, clearly state what's missing

                Current conversation context: {chat_history}
                
                Context from documents: {context}

                Question: {question}
                
                Detailed Answer:"""
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={
                        "k": 15,  # Increased number of retrieved chunks
                        "fetch_k": 30  # Fetch more candidates before selecting top k
                    }
                ),
                memory=self.memory,
                return_source_documents=True,
                verbose=True,
                combine_docs_chain_kwargs={"prompt": prompt}
            )

    def _get_pdf_metadata(self, file_path: str) -> Dict:
        """Extract metadata from PDF file."""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return {
                    'pages': len(reader.pages),
                    'size': os.path.getsize(file_path),
                    'upload_date': datetime.now().isoformat(),
                    'title': os.path.basename(file_path),
                }
        except Exception as e:
            print(f"Error extracting PDF metadata: {str(e)}")
            return {}

    @staticmethod
    def get_all_users() -> List[Dict]:
        """Get list of all users and their profiles."""
        base_dir = "user_data"
        users = []
        
        # Always include default user
        default_profile = {
            "user_id": "default",
            "description": "Default user workspace",
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "tags": []
        }
        users.append(default_profile)
        
        # Check other users
        if os.path.exists(base_dir):
            for user_id in os.listdir(base_dir):
                if user_id == "default":
                    continue  # Skip default user as it's already added
                    
                user_dir = os.path.join(base_dir, user_id)
                profile_file = os.path.join(user_dir, "profile.json")
                docs_dir = os.path.join(user_dir, "docs")
                
                # Strict validation of user directory
                if not all([
                    os.path.isdir(user_dir),
                    os.path.exists(profile_file),
                    os.path.isdir(docs_dir),
                    os.access(user_dir, os.R_OK | os.W_OK),  # Check permissions
                    os.access(profile_file, os.R_OK)
                ]):
                    print(f"Invalid user directory structure for {user_id}")
                    continue
                
                try:
                    with open(profile_file, 'r') as f:
                        profile = json.load(f)
                        # Validate profile structure
                        required_fields = ["description", "created_at", "last_active", "tags"]
                        if all(field in profile for field in required_fields):
                            profile["user_id"] = user_id
                            users.append(profile)
                        else:
                            print(f"Invalid profile structure for user {user_id}")
                except Exception as e:
                    print(f"Error reading profile for user {user_id}: {str(e)}")
        
        # Sort by last active time, most recent first, keeping default user at top
        return ([default_profile] + 
                sorted([u for u in users if u["user_id"] != "default"],
                      key=lambda x: x.get("last_active", ""),
                      reverse=True))

    def update_profile(self, description: str = None, tags: List[str] = None):
        """Update user profile."""
        if description is not None:
            self.profile["description"] = description
        if tags is not None:
            self.profile["tags"] = tags
        self.profile["last_active"] = datetime.now().isoformat()
        self._save_json(self.profile_file, self.profile)

    def get_workspace_stats(self) -> Dict:
        """Get statistics about the user's workspace."""
        total_docs = len([f for f in os.listdir(self.docs_dir) if f.endswith('.pdf')])
        total_size = sum(os.path.getsize(os.path.join(self.docs_dir, f)) 
                        for f in os.listdir(self.docs_dir) if f.endswith('.pdf'))
        total_pages = sum(metadata.get('pages', 0) for metadata in self.metadata.values())
        
        return {
            "total_documents": total_docs,
            "total_size": total_size,
            "total_pages": total_pages,
            "total_queries": len(self.chat_history),
            "total_favorites": len(self.favorites),
            "created_at": self.profile.get("created_at", "Unknown"),
            "last_active": self.profile.get("last_active", "Unknown")
        }

    def process_pdfs(self, force_reload=False):
        """Process PDFs in the user's docs directory and create embeddings."""
        try:
            documents = []
            
            # Load all PDFs from the user's docs directory
            for filename in os.listdir(self.docs_dir):
                if filename.endswith('.pdf'):
                    filepath = os.path.join(self.docs_dir, filename)
                    loader = PDFMinerLoader(filepath)
                    documents.extend(loader.load())

            if not documents:
                raise ValueError(f"No PDF documents found in the docs directory for user {self.user_id}")

            # Split documents into chunks
            splits = self.text_splitter.split_documents(documents)
            print(f"Split documents into {len(splits)} chunks for user {self.user_id}")
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(
                documents=splits,
                embedding=self.embeddings
            )
            
            # Save the vectorstore
            self.vectorstore.save_local(self.db_dir, "docs")
            
            # Initialize QA chain
            self._initialize_qa_chain()
            
            print(f"Created new embeddings and initialized QA chain for user {self.user_id}")
            return len(splits)
        except Exception as e:
            print(f"Error processing PDFs: {str(e)}")
            raise

    def add_document(self, file_path: str, tags: List[str] = None) -> bool:
        """Add a document with metadata and optional tags."""
        try:
            metadata = self._get_pdf_metadata(file_path)
            if tags:
                metadata['tags'] = tags
            
            doc_name = os.path.basename(file_path)
            self.metadata[doc_name] = metadata
            self._save_json(self.metadata_file, self.metadata)
            return True
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            return False

    def get_document_metadata(self, doc_name: str) -> Dict:
        """Get metadata for a specific document."""
        return self.metadata.get(doc_name, {})

    def add_to_favorites(self, question: str, answer: str, sources: List[str]):
        """Add a Q&A pair to favorites."""
        favorite = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
        self.favorites.append(favorite)
        self._save_json(self.favorites_file, self.favorites)

    def get_favorites(self) -> List[Dict]:
        """Get all favorite Q&A pairs."""
        return self.favorites

    def remove_from_favorites(self, index: int):
        """Remove a Q&A pair from favorites."""
        if 0 <= index < len(self.favorites):
            self.favorites.pop(index)
            self._save_json(self.favorites_file, self.favorites)

    def add_to_history(self, question: str, answer: str, sources: List[str]):
        """Add a Q&A pair to chat history."""
        history_item = {
            'question': question,
            'answer': answer,
            'sources': sources,
            'timestamp': datetime.now().isoformat()
        }
        self.chat_history.append(history_item)
        self._save_json(self.history_file, self.chat_history)

    def get_history(self) -> List[Dict]:
        """Get chat history."""
        return self.chat_history

    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        self.memory.clear()  # Clear the conversation memory
        self._save_json(self.history_file, self.chat_history)

    def query(self, question: str) -> dict:
        """Query the RAG system with a question."""
        try:
            if not self.qa_chain:
                raise ValueError(f"Please process PDFs first for user {self.user_id}")

            # Get response
            response = self.qa_chain.invoke({
                "question": question
            })
            
            # Extract source document names
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    if hasattr(doc, "metadata") and "source" in doc.metadata:
                        sources.append(os.path.basename(doc.metadata["source"]))
            
            # Add to history
            self.add_to_history(question, response["answer"], sources)
            
            return {
                "answer": response["answer"],
                "sources": response["source_documents"]
            }
        except Exception as e:
            print(f"Error in query: {str(e)}")
            raise

    def copy_document(self, doc_name: str, target_user_id: str) -> bool:
        """Copy a document to another user's workspace."""
        try:
            source_path = os.path.join(self.docs_dir, doc_name)
            target_dir = os.path.join(self.base_dir, target_user_id, "docs")
            target_path = os.path.join(target_dir, doc_name)
            
            # Create target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the document
            shutil.copy2(source_path, target_path)
            
            # Copy metadata if available
            if doc_name in self.metadata:
                target_system = RAGSystem(target_user_id)
                target_system.metadata[doc_name] = self.metadata[doc_name].copy()
                target_system._save_json(target_system.metadata_file, target_system.metadata)
            
            return True
        except Exception as e:
            print(f"Error copying document: {str(e)}")
            return False

    def delete_user_data(self):
        """Delete all data associated with this user."""
        if self.user_id == "default":
            raise ValueError("Cannot delete the default user")
            
        try:
            # Get the user directory
            user_dir = os.path.join(self.base_dir, self.user_id)
            
            if not os.path.exists(user_dir):
                raise FileNotFoundError(f"User directory not found: {user_dir}")
            
            # Clean up resources
            self.vectorstore = None
            self.qa_chain = None
            self.metadata = {}
            self.chat_history = []
            self.favorites = []
            self.profile = {}
            
            # Delete the user directory and all contents
            shutil.rmtree(user_dir)
            print(f"Deleted all data for user {self.user_id}")
            
            return True
        except Exception as e:
            print(f"Error deleting data for user {self.user_id}: {str(e)}")
            raise ValueError(f"Failed to delete user: {str(e)}")

if __name__ == "__main__":
    # Example usage
    rag = RAGSystem()
    
    # Process PDFs only if there are changes
    print("\nChecking for changes in documents...")
    num_chunks = rag.process_pdfs(force_reload=False)  # Never force reload manually
    print(f"Working with {num_chunks} document chunks")
    
    while True:
        # Get user's question
        question = input("\nEnter your question (or 'quit' to exit): ").strip()
        if question.lower() == 'quit':
            break
            
        # Get response
        response = rag.query(question)
        print("\nAnswer:", response["answer"])
        
        # Show just the list of documents used, with cleaned up names
        used_docs = set()
        for source in response["sources"]:
            if hasattr(source, "metadata"):
                doc_name = os.path.basename(source.metadata["source"])
                # Remove any timestamp and ensure single .pdf extension
                doc_name = doc_name.split(" ", 1)[0]
                if not doc_name.endswith('.pdf'):
                    doc_name += '.pdf'
                elif doc_name.endswith('.pdf.pdf'):
                    doc_name = doc_name[:-4]  # Remove one .pdf if duplicated
                used_docs.add(doc_name)
        
        if used_docs:
            print("\nInformation sourced from:")
            for doc in sorted(used_docs):
                print(f"- {doc}")