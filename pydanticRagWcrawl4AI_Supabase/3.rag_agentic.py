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
from supabase import Client
from typing import List


import os
from dotenv import load_dotenv
from supabase import create_client
from openai import AsyncOpenAI
import asyncio

######  defining the code for similarity search

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
Note: system prompt has now been modified so all tools are used:

The sequence is designed to:
- First get an overview of all available documentation (Tool #2)
- Then search for relevant chunks across all docs (Tool #1)
- Finally, deep dive into specific pages that look promising (Tool #3)

This order makes logical sense because:
- You first want to know what documentation is available
- Then find the most relevant parts quickly using RAG
- Finally, get complete context from the most relevant pages

"""

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

For each user question, follow these steps:
1. First, list all available documentation pages to understand what resources are available.
2. Then, use RAG to find relevant documentation chunks that might answer the question.
3. Finally, for any promising URLs found in step 2, retrieve their full content to ensure you have complete context.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
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
    
"""
2 kinds of tool decorators:
- @agent.tool_plain: for plain text tools (no deps in the tool function)
- @agent.tool: for tools that return a value (use deps in the tool function)

"""
# in pydantic_ai the doc string in the tool tells when and how to use the tool.  
# tool #1 - search for relevant chunks across all docs

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages', # name of the function in the supabase, look at the sql file
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {}  # Empty filter to match all documents
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"


# the below code adds more tools to the agent
# tool #2 - list all available documentation pages

@pydantic_ai_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    Retrieve a list of all available Pydantic AI documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is pydantic_ai_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []
 
# tool #3 deep dive into specific pages that look promising
@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"





###### Now Query the database

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase = create_client(supabase_url, supabase_key)

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=openai_api_key)

# defining the dependencies by calling the dataclass defined earlier - class PydanticAIDeps
deps = PydanticAIDeps(supabase=supabase, openai_client=openai_client)

async def run_query(question: str):
    result = await pydantic_ai_expert.run(question, deps=deps)
    print(result.data)

async def main():
    question = "How do I create a simple Pydantic AI agent?"
    await run_query(question)

if __name__ == "__main__":
    asyncio.run(main())


"""
Notice the output is much better now:  All tools are used and the output is much more detailed.

output:

03:15:17.758 pydantic_ai_expert run prompt=How do I create a simple Pydantic AI agent?
03:15:17.759   preparing model and tools run_step=1
03:15:17.759   model request
03:15:18.809   handle model response
03:15:18.810     running tools=['list_documentation_pages']
03:15:18.881   preparing model and tools run_step=2
03:15:18.881   model request
03:15:20.054   handle model response
03:15:20.055     running tools=['retrieve_relevant_documentation']
03:15:20.630   preparing model and tools run_step=3
03:15:20.631   model request
03:15:21.340   handle model response
03:15:21.341     running tools=['get_page_content']
03:15:21.364   preparing model and tools run_step=4
03:15:21.365   model request
03:15:29.567   handle model response
To create a simple Pydantic AI agent, you can follow the example provided in the documentation. Here’s a summary outlining the process:

1. **Import Necessary Classes**: You'll need `Agent` and `RunContext` from the `pydantic_ai` module.

2. **Create an Agent Instance**: Instantiate your agent by specifying the LLM model you'd like to use. You can also define the types for dependencies and results, and provide a system prompt that instructs the model on how to behave.

3. **Define Tools**: Use the `@agent.tool` decorator to define functions (tools) that the agent can invoke. This allows the agent to perform specific actions based on the user input.

4. **Run the Agent**: You can run the agent using `run_sync`, `run`, or `run_stream` methods, depending on your requirement (synchronous vs. asynchronous execution).

Here is a complete example:

```python
from pydantic_ai import Agent, RunContext

# Create an instance of the agent with specified LLM model
roulette_agent = Agent(
    'openai:gpt-4o',        # Specify the LLM model
    deps_type=int,          # Type for the dependencies this agent will use
    result_type=bool,       # The expected result type
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)

# Define the tool
@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:
    Check if the square is a winner`
    return 'winner' if square == ctx.deps else 'loser'

# Run the agent
success_number = 18
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.data)  # Expected output: True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.data)  # Expected output: False
```

In this example, a roulette agent is created that evaluates winning numbers based on user input. The code defines the agent, its tools, and demonstrates how to run the agent with specific dependencies.

For more detailed information, you can refer to the [Pydantic AI Agents Documentation](https://ai.pydantic.dev/agents/).

"""
