import os
from dotenv import load_dotenv
import agentql
from playwright.sync_api import sync_playwright
import json
import time
from openai import OpenAI  # Import Grok

# Load environment variables
load_dotenv(override=True)

# Initialize API keys
AGENT_API_KEY = os.getenv("AGENT_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")  # Add Grok API key
if not AGENT_API_KEY:
    raise ValueError("AGENT_API_KEY not found in environment variables")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables")

# Initialize Grok client
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Updated query structure to match YouTube's layout
YOUTUBE_CHANNEL_QUERY = """
{
    header {
        title(the channel name)
        subscribers(the subscriber count)
    }
    content {
        videos[] {
            title(the video title)
            views(the view count)
            date(the upload date)
        }
    }
}
"""

def get_grok_summary(data):
    """Get a summary from Grok based on the scraped data."""
    prompt = f"Please summarize the following YouTube channel data:\n{json.dumps(data, indent=2)}"
    completion = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content

def scrape_youtube_channel(channel_url):
    """Scrape YouTube channel data using AgentQL"""
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(
            headless=False,
            slow_mo=100
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        )
        
        page = context.new_page()
        wrapped_page = agentql.wrap(page)
        
        print(f"Navigating to {channel_url}")
        wrapped_page.goto(channel_url, wait_until="networkidle")
        
        # Wait for page to load and content to be visible
        print("Waiting for page to load...")
        time.sleep(5)
        
        # Scroll to load more content
        print("Loading more content...")
        wrapped_page.evaluate("""
            window.scrollTo({
                top: 1000,
                behavior: 'smooth'
            });
        """)
        time.sleep(2)
        
        print("Executing query...")
        response = wrapped_page.query_data(YOUTUBE_CHANNEL_QUERY)
        
        # Debug print to see the response structure
        print("Raw response:", json.dumps(response, indent=2))
        
        # Process the response with safer data access
        channel_data = {
            "channel_info": {
                "title": response.get("header", {}).get("title", "Unknown Channel"),
                "subscribers": response.get("header", {}).get("subscribers", "Unknown")
            },
            "videos": [
                {
                    "title": video.get("title", "Unknown Title"),
                    "views": video.get("views", "0"),
                    "date": video.get("date", "Unknown Date")
                }
                for video in response.get("content", {}).get("videos", [])
            ]
        }
        
        return channel_data
        
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        return {"error": str(e)}
        
    finally:
        try:
            if 'context' in locals():
                context.close()
            if 'browser' in locals():
                browser.close()
            if 'playwright' in locals():
                playwright.stop()
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

def format_json_output(data):
    """Format JSON with proper indentation and encoding"""
    return json.dumps(data, indent=2, ensure_ascii=False)

def main():
    print("\n📺 YouTube Channel Scraper")
    print("=" * 50)
    print(f"Using AgentQL API Key: {AGENT_API_KEY[:5]}...")
    
    while True:
        channel_url = input("\nEnter YouTube channel URL (or 'quit' to exit): \n> ")
        
        if channel_url.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        print("\n🔄 Scraping channel data...")
        data = scrape_youtube_channel(channel_url)
        
        if "error" in data:
            print(f"\n❌ Error: {data['error']}")
        else:
            print("\n✅ Channel Information:")
            print("-" * 50)
            print(format_json_output(data))
            
            # Get summary from Grok
            summary = get_grok_summary(data)
            print("\n📝 Summary:")
            print(summary)

if __name__ == "__main__":
    main()



