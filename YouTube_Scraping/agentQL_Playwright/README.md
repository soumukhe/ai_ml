# YouTube Channel Scraper

A Python script that uses AgentQL to scrape YouTube channel information and video data.

## Features

- Scrapes channel name and subscriber count
- Extracts video information (titles, views, upload dates)
- Clean JSON output format
- Interactive command-line interface
- Configurable through environment variables

## Prerequisites

- Python 3.7+
- AgentQL API key (get one from [AgentQL Developer Portal](https://docs.agentql.com))
- Chrome/Chromium browser

## Installation

1. Clone the repository or download the files.

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Playwright browsers:
   ```bash
   playwright install
   ```

4. Create a `.env` file in the project root and add your AgentQL API key:
   ```plaintext
   AGENT_API_KEY=your_actual_agentql_api_key_here
   ```

## Usage

1. Run the script:
   ```bash
   python main-agent.py
   ```

2. Enter a YouTube channel URL when prompted. For example:
   ```
   Enter YouTube channel URL (or 'quit' to exit): 
   > https://www.youtube.com/@DavidOndrej
   ```

3. The script will output channel information and video data in JSON format.

4. Type 'quit' to exit the program.

### Example Output

✅ Channel Information:
--------------------------------------------------
{
  "channel_info": {
    "title": "David Ondrej",
    "subscribers": "129K"
  },
  "videos": [
    {
      "title": "Sam Altman - The Man Who Owns Silicon Valley",
      "views": "1,231,995",
      "date": "1 year ago"
    },
    {
      "title": "Build Anything With ChatGPT, Here’s How",
      "views": "1,098,889",
      "date": "1 year ago"
    },
    {
      "title": "The Man OpenAI Fears The Most",
      "views": "385,788",
      "date": "1 year ago"
    },


##Summary of the Current Implementation:
- AgentQL: This is the primary library being used to construct queries and scrape data from the YouTube channel.
- Playwright: This library is used to control the browser and navigate to the YouTube channel page, allowing you to interact with the page and extract data.    
