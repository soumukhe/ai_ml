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

system_prompt = """
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation to,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

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
# in this case, the tool is used when the user asks a question.
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


######  Now Query the database

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
Notice that there is only one tool used. only one was defined

output:
03:19:42.506 pydantic_ai_expert run prompt=How do I create a simple Pydantic AI agent?
03:19:42.507   preparing model and tools run_step=1
03:19:42.508   model request
03:19:43.678   handle model response
03:19:43.678     running tools=['retrieve_relevant_documentation']
03:19:44.375   preparing model and tools run_step=2
03:19:44.376   model request
03:19:48.016   handle model response
To create a simple Pydantic AI agent, you can follow this minimal example:

```python
from pydantic_ai import Agent

# Initialize the agent with the specified model and system prompt
agent = Agent(
    'gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.'
)

# Run the agent with a query
result = agent.run_sync('Where does "hello world" come from?')

# Print the response from the agent
print(result.data)
```

This example demonstrates how to create an agent using the Pydantic AI framework, send a query, and print the response. The output from the agent will typically provide an answer related to the inquiry.

For more examples and detailed documentation, you can check the following links:
- [Getting Started with Pydantic AI](https://ai.pydantic.dev/examples/)
- [Pydantic AI API Reference](https://ai.pydantic.dev/api/agent/)


"""
