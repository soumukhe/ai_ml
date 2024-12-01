# 📊 YouTube Shorts Analyzer

This tool analyzes YouTube channels and extracts information about their Shorts videos, including view counts and titles. It uses Playwright for web automation and AgentQL for task management.

## Requirements

- Python 3.7 or higher
- pip (Python package installer)
- Anaconda or Miniconda
- AgentQL API key

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd youtube-shorts-analyzer
```

2. Create and activate a Conda environment:

```bash
conda create -n youtube-shorts python=3.12
conda activate youtube-shorts
```

3. Install required packages:
```bash
conda install pip
pip install -r requirements.txt
```

4. Install Playwright browsers:
```bash
playwright install chromium
```

## Environment Setup

1. Create a .env file in the root directory
2. Add your AgentQL API key to the .env file:
```env
AGENTQL_API_KEY=your_agentql_api_key_here
AGENTQL_LOG_LEVEL=INFO
```

To get an AgentQL API key:
1. Sign up at [AgentQL's website]
2. Navigate to your dashboard
3. Generate a new API key

## Usage

1. Ensure your Conda environment is activated:
```bash
conda activate youtube-shorts
```

2. Run the script:
```bash
python main.py
```

3. When prompted, enter a YouTube channel URL. The URL should be in one of these formats:
   - `https://www.youtube.com/@ChannelName`
   - `https://youtube.com/@ChannelName`

Example:
```bash
Please enter the YouTube channel URL: https://www.youtube.com/@MrBeast
```

## Example Output

```
Channel Information:
Channel Name: MrBeast
Subscribers: 240M subscribers

Shorts Analysis:
Total Shorts Found: 25

Individual Shorts:
1. Title: I Spent 7 Days Buried Alive!
   Views: 54M

2. Title: Would You Swim With Sharks For $100,000?
   Views: 122M

3. Title: Extreme Hide & Seek In A Prison!
   Views: 89M
...
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'playwright'**
   - Solution: Ensure you're in the conda environment (`conda activate youtube-shorts`) and run `pip install -r requirements.txt`

2. **Browser not found error**
   - Solution: Run `playwright install chromium`

3. **AgentQL API Key Error**
   - Solution: Verify your API key is correctly set in the .env file
   - Check if the .env file is in the root directory
   - Ensure the API key is valid and active

4. **Conda environment issues**
   - Solution: Try recreating the environment:
   ```bash
   conda deactivate
   conda env remove -n youtube-shorts
   conda create -n youtube-shorts python=3.12
   conda activate youtube-shorts
   conda install pip
   pip install -r requirements.txt
   ```

5. **Channel Not Found**
   - Solution: Make sure the channel URL is correct
   - Try using the channel's handle (@username) instead of custom URL

6. **No Shorts Found**
   - Solution: Verify the channel actually has Shorts content
   - Try increasing the scroll count in get_shorts_data method

### Performance Issues

If the script is running slowly:
1. Check your internet connection
2. Adjust the sleep timers in the code
3. Reduce the number of scroll iterations if you only need recent shorts

### Browser Issues

If you encounter browser-related problems:
1. Update Playwright: `pip install --upgrade playwright`
2. Reinstall browsers: `playwright install --force`
3. Try running in non-headless mode by changing `headless=True` to `headless=False`

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
