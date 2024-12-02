import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import sys
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def validate_url(url):
    """Validate if the given URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def convert_to_markdown_with_gpt4(content):
    """Convert the scraped content to markdown using GPT-4o."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY not found in .env file")
            return None

        # Get optional configuration from .env
        api_base = os.getenv('OPENAI_API_BASE')
        org_id = os.getenv('OPENAI_ORG_ID')
        
        # Initialize OpenAI client with optional configuration
        client_kwargs = {}
        if api_base:
            client_kwargs['base_url'] = api_base
        if org_id:
            client_kwargs['organization'] = org_id
            
        client = OpenAI(**client_kwargs)  # API key will be automatically read from environment
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o flagship model for complex tasks
            messages=[
                {"role": "system", "content": "You are a helpful assistant that converts web content to clean markdown format."},
                {"role": "user", "content": f"""Please convert the following web content into well-formatted markdown. 
                Make it clean, organized, and easy to read. Preserve all important information while improving readability:

                {content}"""}
            ],
            temperature=0.3,
            max_tokens=4096  # Increased token limit for larger content
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in markdown conversion: {str(e)}")
        return None

def scrape_webpage(url):
    """Scrape the webpage and extract relevant information."""
    try:
        # Get timeout from .env or use default
        timeout = int(os.getenv('REQUESTS_TIMEOUT', 30))
        
        # Send request with a common user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Collect all content in a structured way
        content = {
            "title": soup.title.string if soup.title else "No title found",
            "headers": [header.get_text().strip() for header in soup.find_all(['h1', 'h2', 'h3'])],
            "paragraphs": [para.get_text().strip() for para in soup.find_all('p') if para.get_text().strip()],
            "links": [(link.get_text().strip(), link.get('href')) for link in soup.find_all('a') 
                     if link.get('href') and link.get_text().strip()]
        }

        # Format content for display and GPT-4o
        raw_content = f"""
Title: {content['title']}

Headers:
{chr(10).join(['- ' + header for header in content['headers']])}

Content:
{chr(10).join(['- ' + para[:150] + '...' for para in content['paragraphs']])}

Links:
{chr(10).join(['- ' + text + ': ' + href for text, href in content['links']])}
"""

        print("\n=== Raw Scraped Content ===")
        print(raw_content)

        print("\n=== Converting to Markdown with GPT-4o ===")
        markdown_content = convert_to_markdown_with_gpt4(raw_content)
        
        if markdown_content:
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scraped_content_{timestamp}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            print(f"\nMarkdown content saved to: {filename}")
            
            print("\n=== Converted Markdown Content ===")
            print(markdown_content)
        
    except requests.RequestException as e:
        print(f"Error fetching the webpage: {str(e)}")
        return False
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
    
    return True

def main():
    print("Welcome to the Web Scraper with GPT-4o Markdown Conversion!")
    print("\nNote: Please ensure you have set up your .env file with the required API key.")
    print("You can copy .env.template to .env and add your OpenAI API key.")
    
    while True:
        url = input("\nEnter the URL to scrape (or 'quit' to exit): ").strip()
        
        if url.lower() == 'quit':
            print("Goodbye!")
            sys.exit(0)
            
        if not validate_url(url):
            print("Invalid URL format. Please enter a valid URL (e.g., https://www.example.com)")
            continue
            
        print(f"\nScraping {url}...")
        scrape_webpage(url)
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 