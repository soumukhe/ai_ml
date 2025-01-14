import os
import glob
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv
import hashlib
import json

# Load environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAGSystem:
    def __init__(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Use OpenAI's embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv('OPENAI_API_KEY'),
            model_name="text-embedding-3-small"
        )
        
        # Create or get collection for documents
        self.collection = self.client.get_or_create_collection(
            name="documentation",
            embedding_function=self.embedding_function
        )
        
        # Create or get collection for metadata (stores document hashes)
        self.metadata_collection = self.client.get_or_create_collection(
            name="document_metadata"
        )
        
        # Conversation memory
        self.conversation_history = []
    
    def _get_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _get_stored_hashes(self) -> Dict[str, str]:
        """Get stored document hashes from metadata collection."""
        try:
            result = self.metadata_collection.get(
                ids=["document_hashes"]
            )
            if result and result['documents']:
                return json.loads(result['documents'][0])
            return {}
        except Exception:
            return {}
    
    def _store_hashes(self, hashes: Dict[str, str]):
        """Store document hashes in metadata collection."""
        try:
            self.metadata_collection.delete(ids=["document_hashes"])
        except Exception:
            pass
        
        self.metadata_collection.add(
            documents=[json.dumps(hashes)],
            ids=["document_hashes"],
            metadatas=[{"type": "document_hashes"}]
        )
        
    def load_documents(self, directory: str = "data"):
        """Load and embed markdown documents from the specified directory."""
        markdown_files = glob.glob(f"{directory}/*.md")
        
        # Get current file hashes
        current_hashes = {file: self._get_file_hash(file) for file in markdown_files}
        stored_hashes = self._get_stored_hashes()
        
        # Identify new, modified, and deleted files
        new_or_modified_files = [
            f for f in markdown_files 
            if f not in stored_hashes or stored_hashes[f] != current_hashes[f]
        ]
        deleted_files = [
            f for f in stored_hashes 
            if f not in current_hashes
        ]
        
        if not new_or_modified_files and not deleted_files:
            print("No document changes detected. Using existing embeddings.")
            return
        
        # Remove embeddings for deleted files
        if deleted_files:
            print(f"Removing embeddings for {len(deleted_files)} deleted files...")
            for file in deleted_files:
                try:
                    self.collection.delete(
                        where={"source": file}
                    )
                except Exception as e:
                    print(f"Error removing embeddings for {file}: {e}")
        
        # Add embeddings for new or modified files
        if new_or_modified_files:
            print(f"Processing {len(new_or_modified_files)} new or modified files...")
            for file_path in new_or_modified_files:
                print(f"Processing: {os.path.basename(file_path)}")
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Clean up HTML-like content
                    content = self._clean_markdown(content)
                    
                    # Split content into smaller chunks
                    chunks = self._chunk_text(content, chunk_size=1500)
                    
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
        
        # Store updated hashes
        self._store_hashes(current_hashes)
    
    def _clean_markdown(self, content: str) -> str:
        """Clean up markdown content to better handle code blocks and HTML."""
        # Remove HTML-style links
        import re
        content = re.sub(r'\[(.*?)\]\((.*?)\)', r'\1', content)
        
        # Remove HTML tags but preserve code blocks
        lines = content.split('\n')
        cleaned_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                cleaned_lines.append(line)
                continue
                
            if in_code_block:
                cleaned_lines.append(line)
            else:
                # Remove HTML tags outside code blocks
                cleaned_line = re.sub(r'<[^>]+>', '', line)
                if cleaned_line.strip():
                    cleaned_lines.append(cleaned_line)
        
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

    def query(self, question: str, n_results: int = 5) -> str:
        """Process a question and return an answer using RAG."""
        # Add the question to conversation history
        self.conversation_history.append({"role": "user", "content": question})
        
        # Get relevant documents from ChromaDB
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results,
            include=["metadatas", "documents"]
        )
        
        # Prepare context with source information
        contexts = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source = os.path.basename(metadata['source'])
            contexts.append(f"From {source}:\n{doc}")
        context = "\n\n---\n\n".join(contexts)
        
        # Enhanced system prompt for better responses
        system_prompt = """You are a highly knowledgeable AI assistant with access to specific documentation.
Your role is to:
1. Provide accurate, detailed answers based on the given context
2. When code is present in the context, include it in your response with proper formatting
3. Always preserve code blocks exactly as they appear in the context
4. Clearly indicate which source file the information comes from
5. If the context doesn't contain enough information, acknowledge this and suggest what additional information might be needed
6. Format responses in markdown, especially for code blocks
7. If multiple code snippets are found, combine them logically and explain how they work together"""

        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context from documentation:\n{context}\n\nUser Question: {question}\n\nPlease provide a detailed response based on the above context and any relevant previous conversation history. If there's code in the context, make sure to include it with proper explanation."}
        ]
        
        # Add conversation history for context (last 5 exchanges)
        messages[1:1] = self.conversation_history[-10:-1]  # Insert history before current question
        
        # Get response from OpenAI with updated parameters
        response = openai.chat.completions.create(
            model="gpt-4o",  # OpenAI's flagship model
            messages=messages,
            temperature=0.7,  # Default temperature for balanced creativity and accuracy
            max_tokens=16384,  # Maximum output tokens as per documentation
            top_p=1,         # Default top_p for natural language generation
            presence_penalty=0,  # Default presence penalty
            frequency_penalty=0  # Default frequency penalty
        )
        
        answer = response.choices[0].message.content
        
        # Add the answer to conversation history
        self.conversation_history.append({"role": "assistant", "content": answer})
        
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
