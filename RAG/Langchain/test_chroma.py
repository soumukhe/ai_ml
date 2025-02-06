from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import shutil
from pathlib import Path
import time

# Constants
DOCS_DIR = 'docs'
DOCS_TO_ADD_DIR = 'docs_to_add'
PERSIST_DIRECTORY = 'vector_store'

def test_document_management():
    print("\nStarting document management test...")
    
    # Create directories first
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(DOCS_TO_ADD_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name='thenlper/gte-large')

    # Load documents first
    pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base',
        chunk_size=512,
        chunk_overlap=16
    )

    # Process initial documents
    print("\nProcessing initial documents...")
    doc_chunks = pdf_loader.load_and_split(text_splitter)
    print(f"Initially loaded {len(doc_chunks)} document chunks")

    # List current documents
    print("\nCurrent documents in directory:")
    docs = list(Path(DOCS_DIR).glob('*.pdf'))
    if docs:
        for doc in docs:
            print(f"- {doc.name}")
    else:
        print("No documents in docs directory")

    # Test adding new documents
    print("\nChecking for new documents to add...")
    docs_to_add = list(Path(DOCS_TO_ADD_DIR).glob('*.pdf'))
    if docs_to_add:
        print(f"Found {len(docs_to_add)} new documents to add:")
        for doc in docs_to_add:
            print(f"- {doc.name}")
            # Copy new document to docs directory
            shutil.copy2(doc, DOCS_DIR)
            print(f"Copied {doc.name} to docs directory")
        
        # Load and process all documents
        print("\nProcessing all documents...")
        pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
        all_doc_chunks = pdf_loader.load_and_split(text_splitter)
        print(f"Total document chunks after addition: {len(all_doc_chunks)}")
        
        # Create/Update vector store with all documents
        print("\nCreating vector store with all documents...")
        vector_store = Chroma.from_documents(
            documents=all_doc_chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIRECTORY
        )
        print("Vector store created successfully")
        
        # Test retrieval after addition
        print("\nTesting retrieval after adding documents...")
        retriever = vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 5}
        )
        query = "What is ROCE?"
        docs = retriever.invoke(query)
        print(f"Found {len(docs)} relevant documents for query: {query}")
        
        # Test persistence by creating a new instance
        print("\nTesting persistence by creating new Chroma instance...")
        persisted_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        persisted_retriever = persisted_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': 5}
        )
        persisted_docs = persisted_retriever.invoke(query)
        print(f"Found {len(persisted_docs)} relevant documents from persisted store")
        
        # List final documents
        print("\nFinal documents in directory:")
        for doc in Path(DOCS_DIR).glob('*.pdf'):
            print(f"- {doc.name}")

        # Test delete and re-add functionality
        print("\nTesting delete and re-add functionality...")
        
        # Select a document to delete
        docs = list(Path(DOCS_DIR).glob('*.pdf'))
        if docs:
            doc_to_delete = docs[0]  # Take the first document
            doc_name = doc_to_delete.name
            print(f"\nDeleting document: {doc_name}")
            
            # Save a copy before deleting
            backup_path = Path(DOCS_TO_ADD_DIR) / doc_name
            if not backup_path.exists():
                shutil.copy2(doc_to_delete, backup_path)
                print(f"Backed up {doc_name} to docs_to_add directory")
            
            # Delete the document
            os.remove(doc_to_delete)
            print(f"Deleted {doc_name} from docs directory")
            
            # Process remaining documents
            pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
            remaining_chunks = pdf_loader.load_and_split(text_splitter)
            print(f"Remaining document chunks: {len(remaining_chunks)}")
            
            # Update existing vector store
            print("\nUpdating vector store without deleted document...")
            vector_store = Chroma.from_documents(
                documents=remaining_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            print("Vector store updated successfully")
            
            # Test retrieval after deletion
            print("\nTesting retrieval after deletion...")
            retriever = vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5}
            )
            docs = retriever.invoke(query)
            print(f"Found {len(docs)} relevant documents for query: {query}")
            
            # Re-add the deleted document
            print(f"\nRe-adding document: {doc_name}")
            shutil.copy2(backup_path, DOCS_DIR)
            print(f"Copied {doc_name} back to docs directory")
            
            # Process all documents again
            pdf_loader = PyPDFDirectoryLoader(DOCS_DIR)
            final_chunks = pdf_loader.load_and_split(text_splitter)
            print(f"Final document chunks: {len(final_chunks)}")
            
            # Update vector store with all documents
            print("\nUpdating vector store with re-added document...")
            vector_store = Chroma.from_documents(
                documents=final_chunks,
                embedding=embedding_model,
                persist_directory=PERSIST_DIRECTORY
            )
            print("Vector store updated successfully")
            
            # Test retrieval after re-adding
            print("\nTesting retrieval after re-adding document...")
            retriever = vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5}
            )
            docs = retriever.invoke(query)
            print(f"Found {len(docs)} relevant documents for query: {query}")
            
            # Test persistence again
            print("\nTesting persistence after all operations...")
            final_persisted_store = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_model
            )
            final_persisted_retriever = final_persisted_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': 5}
            )
            final_persisted_docs = final_persisted_retriever.invoke(query)
            print(f"Found {len(final_persisted_docs)} relevant documents from final persisted store")
            
            # List final documents
            print("\nFinal documents in directory:")
            for doc in Path(DOCS_DIR).glob('*.pdf'):
                print(f"- {doc.name}")
        else:
            print("No documents available to test delete functionality")
    else:
        print("No new documents found to add")

if __name__ == "__main__":
    test_document_management() 