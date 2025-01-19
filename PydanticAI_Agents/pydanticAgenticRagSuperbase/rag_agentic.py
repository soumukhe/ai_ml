from __future__ import annotations as _annotations
# forward references in type annotations. must be at the top of the file

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import create_client, Client
from typing import List

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

# defining the dataclass for the dependencies
@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

"""
The sequence is designed to:
- First get an overview of all available PDF documents
- Then search for relevant chunks across all PDFs
- Finally, deep dive into specific pages that look promising

This order makes logical sense because:
- You first want to know what PDF documents are available
- Then find the most relevant parts quickly using RAG
- Finally, get complete context from the most relevant pages
"""

system_prompt = """
You are an expert at analyzing and retrieving information from our PDF document collection.
You have access to all the PDF documents that have been processed and stored in our database.

Your job is to help users find and understand information from these PDF documents.

For each user question, follow these steps:
1. First, list all available PDF documents to understand what resources are available.
2. Then, use RAG to find relevant document chunks that might answer the question.
3. Finally, for any promising pages found in step 2, retrieve their full content to ensure you have complete context.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question.

Always let the user know when you didn't find the answer in the documents or if the search results aren't relevant - be honest.
"""

pdf_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
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

# Tool #1 - search for relevant chunks across all PDFs
@pdf_expert.tool
async def retrieve_relevant_content(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant PDF content chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant content chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_pdf_pages',  # Using the PDF pages matching function
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}  # Empty filter to match all documents
            }
        ).execute()
        
        if not result.data:
            return "No relevant content found in the PDF documents."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']} (Page {doc['page_number']}, {doc['file_name']})

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving content: {e}")
        return f"Error retrieving content: {str(e)}"

# Tool #2 - list all available PDF documents
@pdf_expert.tool
async def list_pdf_documents(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available PDF documents.
    
    Returns:
        List[str]: List of unique PDF file names with their page counts
    """
    try:
        # Query Supabase for unique file names and their page counts
        result = ctx.deps.supabase.from_('pdf_pages') \
            .select('file_name, page_number') \
            .execute()
        
        if not result.data:
            return []
            
        # Process results to get unique files and their page counts
        file_stats = {}
        for doc in result.data:
            file_name = doc['file_name']
            if file_name not in file_stats:
                file_stats[file_name] = set()
            file_stats[file_name].add(doc['page_number'])
        
        # Format the results
        return [f"{file_name} ({len(pages)} pages)" for file_name, pages in sorted(file_stats.items())]
        
    except Exception as e:
        print(f"Error retrieving PDF documents: {e}")
        return []

# Tool #3 - deep dive into specific pages
@pdf_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], file_name: str, page_number: int) -> str:
    """
    Retrieve the full content of a specific PDF page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        file_name: The name of the PDF file
        page_number: The page number to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this page, ordered by chunk number
        result = ctx.deps.supabase.from_('pdf_pages') \
            .select('title, content, metadata') \
            .eq('file_name', file_name) \
            .eq('page_number', page_number) \
            .execute()
        
        if not result.data:
            return f"No content found for {file_name} page {page_number}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title']
        formatted_content = [f"# {page_title}\nFile: {file_name}, Page: {page_number}\n"]
        
        # Sort chunks by their chunk number from metadata
        chunks = sorted(result.data, key=lambda x: x['metadata'].get('chunk_number', 1))
        
        # Add each chunk's content
        for chunk in chunks:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

# Initialize Supabase and OpenAI clients
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)

# Initialize dependencies
deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)

async def run_query(question: str):
    result = await pdf_expert.run(question, deps=deps)
    print(result.data)

async def main():
    #question = "What are the key components of Cisco's AI/ML architecture?"
    question = "Please give me informaton on ROCE"
    await run_query(question)

if __name__ == "__main__":
    asyncio.run(main())