-- Enable the pgvector extension
create extension if not exists vector;

-- Create the pdf_pages table
create table pdf_pages (
    id bigserial primary key,
    file_name varchar not null,
    page_number integer not null,
    title varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate pages for the same file
    unique(file_name, page_number)
);

-- Create an index for better vector similarity search performance
create index on pdf_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_pdf_pages_metadata on pdf_pages using gin (metadata);

-- Create a function to search for pdf pages
create function match_pdf_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
    id bigint,
    file_name varchar,
    page_number integer,
    title varchar,
    content text,
    metadata jsonb,
    similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    file_name,
    page_number,
    title,
    content,
    metadata,
    1 - (pdf_pages.embedding <=> query_embedding) as similarity
  from pdf_pages
  where metadata @> filter
  order by pdf_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Enable RLS on the table
alter table pdf_pages enable row level security;

-- Create a policy that allows anyone to read
create policy "Allow public read access"
  on pdf_pages
  for select
  to public
  using (true);
