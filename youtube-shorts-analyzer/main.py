import os
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import time
from typing import Dict, List
import re
import agentql
import json

# Load environment variables
load_dotenv()

# Verify AgentQL API key is present
AGENTQL_API_KEY = os.getenv('AGENTQL_API_KEY')
if not AGENTQL_API_KEY:
    raise ValueError("AGENTQL_API_KEY not found in environment variables. Please add it to your .env file.")

# Configure AgentQL with API key
agentql.configure(api_key=AGENTQL_API_KEY)

class YoutubeShortsAnalyzer:
    def __init__(self):
        self.browser = None
        self.page = None
        self.playwright = None
        self.channel_stats = {}
        self.shorts_data = []

    def start_browser(self):
        """Initialize browser with AgentQL"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True)
        playwright_page = self.browser.new_page()
        # Wrap Playwright page with AgentQL
        self.page = agentql.wrap(playwright_page)

    def get_channel_stats(self, channel_url: str) -> Dict:
        """Get channel statistics from the channel page using AgentQL"""
        try:
            self.page.goto(channel_url)
            time.sleep(2)

            QUERY = """
            {
                channel_name
                subscriber_count
            }
            """
            
            channel_data = self.page.query_data(QUERY)
            
            print("Raw AgentQL Response:")
            print(json.dumps(channel_data, indent=2))
            
            return {
                "channel_name": channel_data["channel_name"],
                "subscribers": channel_data["subscriber_count"]
            }
        except Exception as e:
            print(f"Failed to get channel stats: {str(e)}")
            raise

    def get_shorts_data(self, channel_url: str) -> List[Dict]:
        """Get data for all shorts from the channel using AgentQL"""
        try:
            # Navigate to shorts tab
            shorts_url = f"{channel_url}/shorts"
            self.page.goto(shorts_url)
            time.sleep(2)

            # Scroll to load more content
            for i in range(5):
                self.page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                time.sleep(1)

            # Use AgentQL query to get shorts data
            QUERY = """
            {
                shorts[] {
                    title
                    views(just the view count with K/M/B suffix)
                    url(get the full URL of the short)
                }
            }
            """
            
            response = self.page.query_data(QUERY)
            return [{"title": short["title"], 
                    "views": short["views"],
                    "url": short["url"]} for short in response["shorts"]]
            
        except Exception as e:
            print(f"Failed to get shorts data: {str(e)}")
            raise

    def analyze_channel(self, channel_url: str):
        """Main function to analyze a YouTube channel's shorts"""
        try:
            self.start_browser()
            
            # Get channel statistics
            self.channel_stats = self.get_channel_stats(channel_url)
            
            # Get shorts data
            self.shorts_data = self.get_shorts_data(channel_url)

        except Exception as e:
            raise e
        finally:
            if self.browser:
                self.browser.close()
                self.browser = None
            if hasattr(self, 'playwright'):
                self.playwright.stop()
                self.playwright = None
            self.page = None

def main():
    while True:
        channel_url = input("\nPlease enter the YouTube channel URL (or 'quit' to exit): ")
        
        if channel_url.lower() in ['quit', 'q', 'exit']:
            print("\nThank you for using YouTube Shorts Analyzer!")
            break
            
        try:
            analyzer = YoutubeShortsAnalyzer()  # Create new instance for each analysis
            analyzer.analyze_channel(channel_url)
        except Exception as e:
            print(f"\nError analyzing channel: {str(e)}")
        
        # Ask if user wants to analyze another channel
        while True:
            continue_analysis = input("\nWould you like to analyze another channel? (yes/no): ").lower()
            if continue_analysis in ['yes', 'y', 'no', 'n']:
                break
            print("Please enter 'yes' or 'no'")
        
        if continue_analysis in ['no', 'n']:
            print("\nThank you for using YouTube Shorts Analyzer!")
            break

if __name__ == "__main__":
    main()
