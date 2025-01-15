import os
import glob
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv
import hashlib
import json
import shutil
import time

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAGSystem:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            print("Creating new RAG system instance...")
            cls._instance = super(RAGSystem, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance and clean up resources."""
        if hasattr(cls, '_instance') and cls._instance is not None:
            try:
                # Clean up ChromaDB client
                if hasattr(cls._instance, 'client') and cls._instance.client is not None:
                    try:
                        # Delete collections first
                        cls._instance.client.delete_collection("documentation")
                        cls._instance.client.delete_collection("chat_history")
                    except Exception as e:
                        print(f"Error deleting collections: {e}")
                    
                    # Reset the client
                    try:
                        cls._instance.client.reset()
                    except Exception as e:
                        print(f"Error resetting client: {e}")
                    
                    cls._instance.client = None
                
                # Clear instance
                cls._instance = None
                print("RAG system instance reset successfully")
            except Exception as e:
                print(f"Error during reset: {e}")
                cls._instance = None
    
    def __init__(self):
        """Initialize the RAG system with in-memory ChromaDB."""
        if self._initialized:
            return
            
        try:
            print("Initializing RAG system...")
            
            # Initialize ChromaDB first
            self.initialize_chroma()
            
            # Initialize conversation history
            self.conversation_history = self._load_chat_history()
            print(f"Loaded {len(self.conversation_history)} messages from chat history")
            
            # Try to load existing embeddings and hashes
            if os.path.exists("./chroma_db/documents.json"):
                print("Found existing embeddings, loading...")
                try:
                    with open("./chroma_db/documents.json", "r") as f:
                        docs_data = json.load(f)
                    
                    if all(key in docs_data for key in ["embeddings", "metadatas", "documents", "ids"]):
                        print(f"Loading {len(docs_data['documents'])} documents into ChromaDB...")
                        self.collection.add(
                            embeddings=docs_data["embeddings"],
                            metadatas=docs_data["metadatas"],
                            documents=docs_data["documents"],
                            ids=docs_data["ids"]
                        )
                        print("Successfully loaded embeddings")
                        self._initialized = True
                        return
                    else:
                        print("Invalid documents.json format")
                except Exception as e:
                    print(f"Error loading embeddings: {str(e)}")
            
            # If we get here, we need to process documents
            print("Will create new embeddings when documents are processed.")
            self._initialized = True
            
        except Exception as e:
            self._initialized = False
            self.__class__.reset_instance()  # Reset instance on initialization failure
            raise Exception(f"Failed to initialize RAG system: {str(e)}")
    
    def initialize_chroma(self):
        """Initialize ChromaDB client and collections."""
        try:
            print("Initializing ChromaDB...")
            
            # Use EphemeralClient instead of PersistentClient
            self.client = chromadb.Client()
            
            # Create new collections
            print("Creating new collections...")
            try:
                self.collection = self.client.create_collection(
                    name="documentation",
                    metadata={"hnsw:space": "cosine"}
                )
                
                self.chat_collection = self.client.create_collection(
                    name="chat_history",
                    metadata={"hnsw:space": "cosine"}
                )
                print("ChromaDB collections created successfully")
                
            except Exception as e:
                print(f"Error creating collections: {e}")
                # Try getting existing collections instead
                try:
                    self.collection = self.client.get_collection("documentation")
                    self.chat_collection = self.client.get_collection("chat_history")
                    print("Retrieved existing collections")
                except Exception as inner_e:
                    print(f"Error getting existing collections: {inner_e}")
                    raise
            
            print("ChromaDB initialized successfully")
            
        except Exception as e:
            print(f"Error in initialize_chroma: {e}")
            raise Exception(f"Failed to initialize ChromaDB: {str(e)}")
    
    def load_from_json(self):
        """Load embeddings from JSON files."""
        try:
            # Load documents collection
            if os.path.exists("./chroma_db/documents.json"):
                print("Found documents.json, loading...")
                with open("./chroma_db/documents.json", "r") as f:
                    docs_data = json.load(f)
                    if not all(key in docs_data for key in ["embeddings", "metadatas", "documents", "ids"]):
                        print("Missing required keys in documents.json")
                        return
                    
                    if len(docs_data["documents"]) == 0:
                        print("No documents found in documents.json")
                        return
                    
                    print(f"Adding {len(docs_data['documents'])} documents to collection...")
                    try:
                        self.collection.add(
                            embeddings=docs_data["embeddings"],
                            metadatas=docs_data["metadatas"],
                            documents=docs_data["documents"],
                            ids=docs_data["ids"]
                        )
                        print("Successfully added documents to collection")
                    except Exception as e:
                        print(f"Error adding documents to collection: {str(e)}")
                        raise
            else:
                print("No documents.json found")
                return
            
            # Verify collection has data
            try:
                doc_count = len(self.collection.get()["documents"])
                print(f"Verified collection has {doc_count} documents")
                if doc_count == 0:
                    print("Warning: Collection is empty after loading")
            except Exception as e:
                print(f"Error verifying collection data: {str(e)}")
                raise
                
        except Exception as e:
            print(f"Error loading from JSON: {str(e)}")
            raise Exception(f"Failed to load from JSON: {str(e)}")
    
    def save_to_json(self):
        """Save embeddings and metadata to JSON files."""
        try:
            os.makedirs("./chroma_db", exist_ok=True)
            
            # Save documents collection
            docs = self.collection.get()
            with open("./chroma_db/documents.json", "w") as f:
                json.dump(docs, f)
            
            print("Successfully saved embeddings to JSON")
            
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")
            raise Exception(f"Failed to save to JSON: {str(e)}")
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_stored_hashes(self) -> Dict[str, str]:
        """Get stored document hashes from JSON file."""
        try:
            hash_file = "./chroma_db/document_hashes.json"
            if os.path.exists(hash_file):
                print(f"Loading hashes from {hash_file}")
                with open(hash_file, 'r') as f:
                    hashes = json.load(f)
                    print(f"Loaded {len(hashes)} file hashes")
                    return hashes
            print("No hash file found")
            return {}
        except Exception as e:
            print(f"Error loading hashes: {e}")
            return {}
    
    def _store_hashes(self, hashes: Dict[str, str]):
        """Store document hashes in JSON file."""
        try:
            hash_file = "./chroma_db/document_hashes.json"
            print(f"Saving {len(hashes)} hashes to {hash_file}")
            with open(hash_file, 'w') as f:
                json.dump(hashes, f)
            print("Successfully saved hashes")
        except Exception as e:
            print(f"Warning: Could not store hashes: {e}")
    
    def load_documents(self, directory: str = "data", include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        """Load and embed markdown documents from the specified directory."""
        print(f"\nScanning for markdown files in {directory}...")
        
        # First check if we have documents in ChromaDB
        try:
            doc_count = len(self.collection.get()["documents"])
            if doc_count == 0:
                print("No documents in ChromaDB. Processing all files...")
                need_full_process = True
            else:
                print(f"Found {doc_count} documents in ChromaDB")
                need_full_process = False
        except Exception:
            print("Error checking ChromaDB. Processing all files...")
            need_full_process = True
        
        # Build list of files to process
        markdown_files = []
        
        # Use simple glob to get all markdown files in the root data directory
        markdown_files = glob.glob(f"{directory}/*.md")
        
        print(f"Found {len(markdown_files)} markdown files")
        for f in markdown_files:
            print(f"  - {os.path.relpath(f, directory)}")
        
        # Get current file hashes and save them immediately
        print("\nCalculating file hashes...")
        current_hashes = {file: self._get_file_hash(file) for file in markdown_files}
        print(f"Calculated {len(current_hashes)} hashes")
        
        # Save hashes immediately after calculation
        self._store_hashes(current_hashes)
        
        print("\nLoading stored hashes for comparison...")
        stored_hashes = self._get_stored_hashes()
        print(f"Loaded {len(stored_hashes)} stored hashes")
        
        # If ChromaDB is empty, process all files regardless of hashes
        if need_full_process:
            print("\nProcessing all files due to empty ChromaDB...")
            new_or_modified_files = markdown_files
            deleted_files = []
        else:
            # Identify new, modified, and deleted files
            new_or_modified_files = [
                f for f in markdown_files 
                if f not in stored_hashes or stored_hashes[f] != current_hashes[f]
            ]
            deleted_files = [
                f for f in stored_hashes 
                if f not in current_hashes
            ]
        
        print(f"\nFiles to process: {len(new_or_modified_files)}")
        for f in new_or_modified_files:
            print(f"  - {os.path.basename(f)}")
        
        print(f"\nDeleted files: {len(deleted_files)}")
        for f in deleted_files:
            print(f"  - {os.path.basename(f)}")
        
        if not need_full_process and not new_or_modified_files and not deleted_files:
            print("\nNo document changes detected and ChromaDB has documents. Using existing embeddings.")
            return
        
        # Process files
        if new_or_modified_files:
            print(f"\nProcessing {len(new_or_modified_files)} files...")
            for file_path in new_or_modified_files:
                print(f"Processing: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Clean up HTML-like content
                    content = self._clean_markdown(content)
                    
                    # Split content into chunks with larger chunk size
                    chunks = self._chunk_text(content, chunk_size=2500)
                    
                    # Remove old embeddings for this file if it was modified
                    try:
                        self.collection.delete(
                            where={"source": file_path}
                        )
                    except Exception:
                        pass
                    
                    # Add new embeddings
                    self.collection.add(
                        documents=chunks,
                        ids=[f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))],
                        metadatas=[{
                            "source": file_path,
                            "chunk_index": i,
                            "total_chunks": len(chunks)
                        } for i in range(len(chunks))]
                    )
                # Save after each file is processed
                self.save_to_json()
        
        # Store updated hashes
        self._store_hashes(current_hashes)
        
        # Final save to ensure everything is persisted
        self.save_to_json()
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content while preserving important documentation elements.
        
        This function:
        1. Preserves code blocks and their content
        2. Maintains markdown-style links
        3. Keeps headers and formatting
        4. Removes only problematic HTML elements
        """
        import re
        
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False
        code_block_lang = None
        
        for line in lines:
            # Handle code blocks
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    # Extract language if specified
                    code_block_lang = line.strip()[3:]
                cleaned_lines.append(line)
                continue
                
            if in_code_block:
                # Preserve code blocks exactly as they are
                cleaned_lines.append(line)
            else:
                # Outside code blocks:
                # 1. Remove HTML comments
                line = re.sub(r'<!--.*?-->', '', line)
                
                # 2. Preserve markdown links while cleaning HTML-style links
                line = re.sub(r'<a\s+href="([^"]*)"[^>]*>(.*?)</a>', r'[\2](\1)', line)
                
                # 3. Remove other HTML tags but preserve their content
                line = re.sub(r'<[^>]+>', '', line)
                
                # 4. Preserve headers and formatting
                if re.match(r'^#{1,6}\s+', line) or re.match(r'^[-*]\s+', line):
                    cleaned_lines.append(line)
                else:
                    # 5. Clean up extra whitespace
                    line = ' '.join(line.split())
                    if line:
                        cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)

    def _chunk_text(self, text: str, chunk_size: int = 1500) -> List[str]:
        """Split text into smaller chunks, trying to preserve code blocks intact."""
        chunks = []
        current_chunk = []
        current_size = 0
        in_code_block = False
        code_block_content = []
        
        for line in text.split('\n'):
            # Handle code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                if in_code_block:
                    # Starting a new code block
                    code_block_content = [line]
                else:
                    # Ending a code block
                    code_block_content.append(line)
                    block = '\n'.join(code_block_content)
                    if current_size + len(block) > chunk_size * 2:
                        # If code block is too large, store it separately
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                        chunks.append(block)
                        current_chunk = []
                        current_size = 0
                    else:
                        current_chunk.extend(code_block_content)
                        current_size += len(block)
                continue
            
            if in_code_block:
                code_block_content.append(line)
            else:
                line_size = len(line) + 1  # +1 for newline
                if current_size + line_size > chunk_size and current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_size = line_size
                else:
                    current_chunk.append(line)
                    current_size += line_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks

    def _save_chat_history(self):
        """Save chat history to a JSON file."""
        try:
            if not os.path.exists("./chroma_db"):
                os.makedirs("./chroma_db", exist_ok=True)
                os.chmod("./chroma_db", 0o777)
            
            with open("./chroma_db/chat_history.json", "w", encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print("Chat history saved successfully")
        except Exception as e:
            print(f"Warning: Could not save chat history: {e}")
    
    def _load_chat_history(self) -> List[Dict[str, str]]:
        """Load chat history from JSON file."""
        try:
            if os.path.exists("./chroma_db/chat_history.json"):
                with open("./chroma_db/chat_history.json", "r", encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load chat history: {e}")
        return []

    def query(self, user_query: str) -> str:
        """Query the RAG system with a user question."""
        # Get relevant documents from ChromaDB
        results = self.collection.query(
            query_texts=[user_query],
            n_results=15,
            include=["documents", "metadatas"]
        )
        
        # Extract relevant context and organize it
        contexts = results["documents"][0]
        context = "\n\n---\n\n".join(contexts)
        
        # Create system message with explicit markdown formatting requirements
        system_message = {
            "role": "system", 
            "content": """You are a technical documentation assistant that provides well-structured, detailed responses using markdown formatting.

Your responses MUST follow this exact structure:

## Overview
[Provide a brief summary of the main concepts]

## Code Example
```python
[Include any relevant code examples from the documentation, preserving exact formatting]
```

## Components and Technologies
[List and explain each component/technology mentioned]
- Component 1: [Explanation]
- Component 2: [Explanation]
[etc.]

## How It Works
[Explain how the components work together and how to use the code]

## Technical Details
[Include any specifications, requirements, or additional technical information]

## Additional Information
[Any other relevant details from the documentation]

Rules:
1. ONLY use information from the provided documentation
2. If information isn't in the documentation, say "The provided documentation does not contain this information"
3. NEVER use your own knowledge
4. Format all technical terms with backticks
5. Use bullet points for lists
6. Include all code examples EXACTLY as shown in the documentation
7. Maintain consistent header formatting"""
        }
        
        # Create context message with the query
        context_message = {
            "role": "user",
            "content": f'''Here is the ONLY documentation you may use to answer:

{context}

Using ONLY this documentation above, answer this question: {user_query}

You MUST:
1. Format your response using the exact structure specified
2. Include any relevant code examples in the Code Example section
3. Keep code formatting exactly as shown in the documentation
4. If a section would be empty, include the section header and state "No information available in the documentation."

Remember: If the information isn't in the documentation provided, say "The provided documentation does not contain this information" rather than using your own knowledge.'''
        }
        
        # Initialize OpenAI client
        client = openai.OpenAI()
        
        # Get response from OpenAI
        messages = [system_message]
        
        # Add recent conversation history if available
        if len(self.conversation_history) > 0:
            messages.extend(self.conversation_history[-2:])
        
        messages.append(context_message)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            response_format={"type": "text"}
        )
        
        # Extract and format the response
        answer = response.choices[0].message.content
        
        # Update conversation history
        self.conversation_history.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": answer}
        ])
        
        # Save updated chat history
        self._save_chat_history()
        
        return answer

    def format_response(self, answer: str) -> str:
        """Format the response in a nice way."""
        separator = "=" * 80
        formatted_response = f"""
{separator}
📚 AI Assistant Response:
{separator}

{answer}

{separator}
"""
        return formatted_response

    def verify_rag_system(self):
        """Verify the RAG system is properly initialized with documents."""
        try:
            # Check if we have documents in ChromaDB
            doc_count = len(self.collection.get()["documents"])
            if doc_count > 0:
                print(f"System ready with {doc_count} document chunks")
                return
            
            # If we get here, we need to process documents
            print("No documents in ChromaDB. Processing...")
            self.load_documents()
            
        except Exception as e:
            print(f"Error in verify_rag_system: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Load documents
    print("Loading and embedding documents...")
    rag.load_documents()
    
    # Interactive query loop
    print("\nRAG System Ready! Type 'exit' to quit.")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() == 'exit':
            break
            
        answer = rag.query(question)
        print(rag.format_response(answer))
