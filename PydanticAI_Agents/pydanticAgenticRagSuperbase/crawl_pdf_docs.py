import os
import sys
import json
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv
import PyPDF2
from pathlib import Path

from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

"""
This code processes PDF files from a directory and stores them in Supabase.

The flow is as follows:

main()
  ├── get_pdf_files()
  │     └── Return list of PDF files
  ├── process_pdfs_parallel(pdf_files)
  │     ├── Create semaphore
  │     ├── process_pdf (nested)
  │     │     ├── Extract PDF text and metadata
  │     │     ├── process_and_store_pages(file_name, pages)
  │     │     │     ├── chunk_text(text)
  │     │     │     ├── process_chunk(chunk, page_num, file_name)
  │     │     │     │     ├── get_title_and_summary(chunk)
  │     │     │     │     ├── get_embedding(chunk)
  │     │     │     │     └── Return ProcessedPage
  │     │     │     ├── insert_page(page)
  │     │     │     └── Repeat for all pages
  │     │     └── Repeat for all PDFs
  │     └── Gather all tasks
  └── Exit program
"""

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedPage:
    file_name: str
    page_number: int
    title: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to break at a paragraph
        chunk = text[start:end]
        if '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end

    return chunks

async def get_title_and_summary(chunk: str) -> Dict[str, str]:
    """Extract title using GPT-4."""
    system_prompt = """You are an AI that extracts titles from PDF content chunks.
    Return a JSON object with a 'title' key.
    Create a descriptive title that summarizes the main topic of this chunk.
    Keep the title concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Content:\n{chunk[:1000]}..."}
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title: {e}")
        return {"title": "Untitled Section"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, page_number: int, file_name: str) -> ProcessedPage:
    """Process a single chunk of text from a PDF page."""
    # Get title
    extracted = await get_title_and_summary(chunk)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pdf_document",
        "chunk_size": len(chunk),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "file_path": str(file_name)
    }
    
    return ProcessedPage(
        file_name=str(file_name),
        page_number=page_number,
        title=extracted['title'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_page(page: ProcessedPage):
    """Insert a processed page into Supabase."""
    try:
        data = {
            "file_name": page.file_name,
            "page_number": page.page_number,
            "title": page.title,
            "content": page.content,
            "metadata": page.metadata,
            "embedding": page.embedding
        }
        
        result = supabase.table("pdf_pages").insert(data).execute()
        print(f"Inserted page {page.page_number} from {page.file_name}")
        return result
    except Exception as e:
        print(f"Error inserting page: {e}")
        return None

async def process_and_store_pages(file_name: str, pages: List[str]):
    """Process PDF pages and store in Supabase."""
    try:
        total_pages = len(pages)
        for page_num, page_text in enumerate(pages, 1):
            # Split page into chunks if it's too long
            chunks = chunk_text(page_text) if len(page_text) > 5000 else [page_text]
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                processed_page = await process_chunk(
                    chunk, 
                    page_number=page_num,
                    file_name=file_name
                )
                # Add chunk number to metadata
                processed_page.metadata["chunk_number"] = i + 1
                processed_page.metadata["total_chunks"] = len(chunks)
                await insert_page(processed_page)
                print(f"Stored page {page_num}/{total_pages} (chunk {i+1}/{len(chunks)}) from {file_name}")
                sys.stdout.flush()
                
        print(f"Successfully processed and stored document: {file_name}")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error processing document {file_name}: {e}")
        sys.stdout.flush()

def get_pdf_files() -> List[Path]:
    """Get all PDF files from the pdf_docs directory."""
    pdf_dir = Path("pdf_docs")
    if not pdf_dir.exists():
        print("pdf_docs directory not found")
        return []
    
    return list(pdf_dir.glob("*.pdf"))

async def process_pdfs_parallel(pdf_files: List[Path], max_concurrent: int = 3):
    """Process multiple PDFs in parallel with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_pdf(pdf_path: Path):
        async with semaphore:
            try:
                print(f"Processing {pdf_path.name}...")  # Progress tracking output
                sys.stdout.flush()  # Ensure output is sent immediately
                
                # Extract text from PDF
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    pages = []
                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text()
                        if text.strip():  # Only include non-empty pages
                            pages.append(text)
                            print(f"Extracted page {page_num} from {pdf_path.name}")  # Progress tracking
                            sys.stdout.flush()
                
                if pages:
                    await process_and_store_pages(pdf_path.name, pages)
                    print(f"Completed processing {pdf_path.name}")  # Progress tracking
                    sys.stdout.flush()
                else:
                    print(f"No text content found in {pdf_path.name}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                sys.stdout.flush()
    
    # Process all PDFs in parallel with limited concurrency
    await asyncio.gather(*[process_pdf(pdf) for pdf in pdf_files])

async def main():
    # Get PDF files
    pdf_files = get_pdf_files()
    if not pdf_files:
        print("No PDF files found to process")
        sys.stdout.flush()
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    sys.stdout.flush()
    
    # Process files
    await process_pdfs_parallel(pdf_files)
    print("All documents processed successfully")
    sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(main())