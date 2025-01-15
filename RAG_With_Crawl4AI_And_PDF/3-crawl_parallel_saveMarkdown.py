import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree
import pathlib
from urllib.parse import urlparse

"""
This time it saves the markdown to a file in the data directory.
This script is designed to crawl a list of URLs in parallel, with session reuse for each URL. 
It uses the Crawl4AI library to perform the crawling and markdown generation.

Example URLs to crawl:
1. FastAPI Documentation: https://fastapi.tiangolo.com/tutorial/
2. Python Requests Documentation: https://requests.readthedocs.io/en/latest/
3. SQLAlchemy Tutorial: https://docs.sqlalchemy.org/en/20/tutorial/
4. Pandas Getting Started: https://pandas.pydata.org/docs/getting_started/
5. Flask Documentation: https://flask.palletsprojects.com/en/3.0.x/

These documentation sites are well-structured and suitable for crawling.

https://docs.crawl4ai.com/advanced/multi-url-crawling/
- Uses the sitemap (https://ai.pydantic.dev/sitemap.xml) to get the URLs.

"""

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# Append parent directory to system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== Parallel Crawling with Browser Reuse + Memory Check ===")

    # We'll keep track of peak memory usage across all tasks
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Minimal browser config
    browser_config = BrowserConfig(
        headless=True,
        verbose=True,   # Enable verbose logging
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS
    )

    # Create the crawler instance
    print("Creating crawler instance...")
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    print("Crawler started successfully")

    # Create data directory if it doesn't exist
    data_dir = pathlib.Path("data")  # Use relative path
    data_dir.mkdir(exist_ok=True)
    print(f"Using data directory: {data_dir.absolute()}")

    try:
        # We'll chunk the URLs in batches of 'max_concurrent'
        success_count = 0
        fail_count = 0
        
        print(f"Processing {len(urls)} URLs in batches of {max_concurrent}")
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                print(f"Preparing to crawl: {url}")
                # Unique session_id per concurrent sub-task
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # Check memory usage prior to launching tasks
            log_memory(prefix=f"Before batch {i//max_concurrent + 1}: ")

            # Gather results with timeout
            print(f"Starting batch {i//max_concurrent + 1}...")
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=300  # 5 minute timeout for entire batch
                )
            except asyncio.TimeoutError:
                print(f"Batch {i//max_concurrent + 1} timed out")
                continue

            print(f"Completed batch {i//max_concurrent + 1}")

            # Check memory usage after tasks complete
            log_memory(prefix=f"After batch {i//max_concurrent + 1}: ")

            # Evaluate results
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"Error crawling {url}: {result}")
                    fail_count += 1
                elif result.success:
                    success_count += 1
                    
                    # Generate a safe filename from the URL
                    parsed_url = urlparse(url)
                    safe_filename = parsed_url.path.strip('/').replace('/', '_') or 'index'
                    if not safe_filename.endswith('.md'):
                        safe_filename += '.md'
                    file_path = data_dir / safe_filename
                    
                    # Save the markdown content
                    try:
                        print(f"Saving markdown to: {file_path}")
                        if hasattr(result.markdown_v2, 'raw_markdown'):
                            file_path.write_text(result.markdown_v2.raw_markdown)
                            print(f"Successfully saved markdown to: {file_path}")
                        else:
                            print(f"No markdown content available for {url}")
                            fail_count += 1
                    except Exception as e:
                        print(f"Error saving markdown for {url}: {e}")
                        fail_count += 1
                else:
                    print(f"Crawling failed for {url}: {result}")
                    fail_count += 1

        print(f"\nSummary:")
        print(f"  - Successfully crawled: {success_count}")
        print(f"  - Failed: {fail_count}")

    finally:
        print("\nClosing crawler...")
        await crawler.close()
        # Final memory log
        log_memory(prefix="Final: ")
        print(f"\nPeak memory usage (MB): {peak_memory // (1024 * 1024)}")

def get_pydantic_ai_docs_urls():
    """
    Fetches URLs from either a sitemap or a regular website URL.
    Supports both sitemap.xml URLs and regular website URLs.
    
    Example URLs:
    - Sitemap: https://ai.pydantic.dev/sitemap.xml
    - Regular: https://fastapi.tiangolo.com/tutorial/
    
    Returns:
        List[str]: List of URLs
    """            
    # This will be replaced by the streamlit app
    sitemap_url = ""
    
    try:
        if not sitemap_url:
            print("No URL provided")
            return []
            
        print(f"Processing URL: {sitemap_url}")
        # Check if it's a sitemap URL
        if sitemap_url.endswith('sitemap.xml'):
            print("Processing sitemap URL...")
            response = requests.get(sitemap_url)
            response.raise_for_status()
            
            # Parse the XML
            root = ElementTree.fromstring(response.content)
            
            # Extract all URLs from the sitemap
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
        else:
            # Handle regular website URL
            print("Processing regular website URL...")
            # Make sure URL has proper scheme
            if not sitemap_url.startswith(('http://', 'https://')):
                sitemap_url = f"https://{sitemap_url}"
                
            # Remove any trailing slashes for consistency
            sitemap_url = sitemap_url.rstrip('/')
            
            # Try to find sitemap first
            base_url = '/'.join(sitemap_url.split('/')[:3])  # Get domain part
            possible_sitemaps = [
                f"{base_url}/sitemap.xml",
                f"{base_url}/sitemap_index.xml",
                f"{base_url}/sitemap/sitemap.xml"
            ]
            
            urls = []
            sitemap_found = False
            
            # Try each possible sitemap location
            for possible_sitemap in possible_sitemaps:
                try:
                    print(f"Trying sitemap: {possible_sitemap}")
                    response = requests.get(possible_sitemap)
                    if response.status_code == 200 and 'xml' in response.headers.get('content-type', ''):
                        root = ElementTree.fromstring(response.content)
                        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
                        sitemap_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
                        if sitemap_urls:
                            print(f"Found sitemap with {len(sitemap_urls)} URLs")
                            urls.extend(sitemap_urls)
                            sitemap_found = True
                            break
                except Exception as e:
                    print(f"Error checking sitemap {possible_sitemap}: {e}")
                    continue
            
            if not sitemap_found:
                print("No sitemap found, crawling from provided URL...")
                # Get the page content
                response = requests.get(sitemap_url)
                response.raise_for_status()
                
                # Use BeautifulSoup to extract links
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract all links that start with the base URL
                base_path = '/'.join(sitemap_url.split('/')[:-1]) + '/'
                for link in soup.find_all('a'):
                    href = link.get('href')
                    if href:
                        # Convert relative URLs to absolute
                        if href.startswith('/'):
                            href = base_url + href
                        elif not href.startswith(('http://', 'https://')):
                            href = base_path + href
                        
                        # Only include URLs from the same domain and path
                        if href.startswith(base_path):
                            urls.append(href)
                
                # Add the original URL if not already included
                if sitemap_url not in urls:
                    urls.append(sitemap_url)
                
                # Remove duplicates while preserving order
                urls = list(dict.fromkeys(urls))
            
        print(f"Found {len(urls)} URLs to process")
        return urls
        
    except Exception as e:
        print(f"Error fetching URLs: {e}")
        return []

async def main():
    urls = get_pydantic_ai_docs_urls()
    if urls:
        print(f"Found {len(urls)} URLs to crawl")
        await crawl_parallel(urls, max_concurrent=5)
    else:
        print("No URLs found to crawl")    

if __name__ == "__main__":
    asyncio.run(main())