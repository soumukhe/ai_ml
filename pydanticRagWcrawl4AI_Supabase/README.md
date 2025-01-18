# pydanticRagWcrawl4AI_Supabase

A Python project that demonstrates how to build a RAG (Retrieval Augmented Generation) system by crawling documentation using Crawl4AI, storing it in Supabase with vector embeddings, and implementing both non-agentic and agentic approaches using Pydantic AI.

## Overview

This project demonstrates how to build a RAG (Retrieval Augmented Generation) system using Pydantic AI and Supabase. It includes both non-agentic and agentic approaches to RAG, along with utilities for crawling documentation and setting up the required infrastructure. The setup uses a local Supabase instance running via docker-compose for development and testing.

## Project Structure

- `1.crawl_pydantic_ai_docs.py`: Crawls the Pydantic AI documentation website using Crawl4AI, chunks the content, generates embeddings, and stores them in Supabase
- `2.rag_non_agentic.py`: Implements a simple RAG system using a single tool for document retrieval
- `3.rag_agentic.py`: Implements an advanced RAG system using three tools with a modified system prompt to ensure comprehensive document retrieval
- `generateSupabaseSecret.py`: Utility to generate Supabase secrets (JWT, service role key, anon key)
- `site_pages.sql`: SQL schema for Supabase database setup
- `testSupbaseCreds.py`: Quick verification tool for Supabase credentials
- `references.md`: Additional context and references

## Setup

### 1. Supabase Setup (Local with Docker Compose)

This project uses a local Supabase instance running in Docker containers for development and testing. This approach provides full control over the database and makes it easier to develop and test without external dependencies.

1. Start Supabase locally:
```bash
docker compose up -d
```

2. Generate required secrets:
```bash
python generateSupabaseSecret.py
```
This will generate:
- JWT_SECRET
- SERVICE_ROLE_KEY
- ANON_KEY

3. Create the database schema:
- Execute the contents of `site_pages.sql` in your local Supabase instance
- This creates:
  - The `site_pages` table with vector support
  - Necessary indexes for efficient retrieval
  - Row level security policies

4. Verify your Supabase setup:
```bash
python testSupbaseCreds.py
```

### 2. Environment Variables

Create a `.env` file with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_role_key
OPENAI_API_KEY=your_openai_api_key
LLM_MODEL=gpt-4o-mini  # or your preferred model
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Crawl Documentation

First, crawl and process the Pydantic AI documentation:
```bash
python 1.crawl_pydantic_ai_docs.py
```

This script:
- Crawls the Pydantic AI documentation
- Chunks the content into manageable pieces
- Generates embeddings using OpenAI
- Stores everything in Supabase

### 2. Non-Agentic RAG

Run the simple RAG implementation:
```bash
python 2.rag_non_agentic.py
```

Features:
- Single tool for document retrieval
- Direct RAG implementation
- Suitable for simple queries

### 3. Agentic RAG

Run the advanced RAG implementation:
```bash
python 3.rag_agentic.py
```

Features:
- Three tools working together:
  1. `list_documentation_pages`: Lists all available documentation
  2. `retrieve_relevant_documentation`: RAG-based retrieval
  3. `get_page_content`: Retrieves full page content
- Modified system prompt to ensure all tools are utilized
- More comprehensive document retrieval

## Tool Descriptions

### Tool #1: retrieve_relevant_documentation
- Purpose: Search for relevant chunks across all documentation
- Uses RAG with embeddings
- Returns top 5 most relevant documentation chunks

### Tool #2: list_documentation_pages
- Purpose: Get overview of all available documentation
- Lists all unique URLs in the database
- Helps understand available resources

### Tool #3: get_page_content
- Purpose: Deep dive into specific pages
- Retrieves full content of specific URLs
- Combines all chunks for complete context

## References

For more information and context, see:
- [Crawl4AI Agent Tutorial](https://www.youtube.com/watch?v=_R-ff4ZMLC8)
- [Ottomator Agents Repository](https://github.com/coleam00/ottomator-agents/tree/main/crawl4AI-agent)

### Supabase Self-Hosting
For local development:
- [Supabase Self-Hosting Guide](https://supabase.com/docs/guides/self-hosting/docker)

## Troubleshooting

### Supabase Database Reset
To reset the database:
```sql
-- Remove all records
DELETE FROM site_pages;

-- Or delete entire table
DROP TABLE site_pages;
```

#### For complete sql db reset:
```sql
# Drop old tables

-- First disable RLS
ALTER TABLE IF EXISTS pdf_pages DISABLE ROW LEVEL SECURITY;

-- Drop the policy
DROP POLICY IF EXISTS "Allow public read access" ON pdf_pages;

-- Drop the function
DROP FUNCTION IF EXISTS match_pdf_pages;

-- Drop the indexes
DROP INDEX IF EXISTS idx_pdf_pages_metadata;
DROP INDEX IF EXISTS pdf_pages_embedding_idx;

-- Drop the table
DROP TABLE IF EXISTS pdf_pages;

-- Drop the extension (optional - you might want to keep this if other tables use it)
-- DROP EXTENSION IF EXISTS vector;
```

### Docker Commands
```bash
# Stop and remove containers with volumes
docker compose down -v

# Remove database data
rm -rf volumes/db/data/

# Recreate and start containers
docker compose up -d
```

Note: Simply restarting containers (`docker compose restart`) is not sufficient for credential changes. Full recreation is required.

## Analytics Configuration

To disable analytics functionality in Supabase:
```toml
[analytics]
enabled = false
```

## Testing Supabase API

Quick curl test:
```bash
curl -X GET "http://localhost:8000/rest/v1/site_pages" \
  -H "Authorization: Bearer <supbase-service-key>" \
  -H "apikey: <supbase-anon-key>"
``` 
