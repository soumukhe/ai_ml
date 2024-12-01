# 📊 YouTube Shorts Analyzer

A powerful tool that analyzes YouTube channels and extracts information about their Shorts videos, featuring a modern web interface with a card-based layout for easy viewing and copying of video titles.

## Features

- 🌐 Modern Web Interface with Streamlit
- 📊 Channel Statistics Dashboard
- 📱 YouTube Shorts Data Extraction
- 🎨 Card-Based Grid Layout
- 📋 One-Click Copy for Video Titles with Toast Notifications
- 👁️ View Count Analysis with Visual Indicators
- 💻 Background Server Mode

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

### Web Interface (Recommended)

1. Ensure your Conda environment is activated:
```bash
conda activate youtube-shorts
```

2. Launch the app in background mode:
```bash
python streamlit_app.py
```

The app will:
- Start automatically in your default browser
- Run the server in the background
- Return control to your terminal
- Display the server URL

3. Interface Features:
   - Clean, modern dashboard layout
   - Channel metrics display
   - 3-column grid of video cards
   - Copy buttons with confirmation toasts
   - View counts with visual indicators

4. To stop the server:
```bash
pkill -f streamlit
```

### Interface Layout

```
📊 YouTube Shorts Analyzer
├── Channel Information
│   ├── Channel Name [Metric]
│   └── Subscriber Count [Metric]
│
└── Shorts Analysis Grid
    ├── Card 1
    │   ├── Title
    │   ├── 👁️ View Count
    │   └── 📋 Copy Button
    ├── Card 2
    │   ├── Title
    │   ├── 👁️ View Count
    │   └── 📋 Copy Button
    └── Card 3
        ├── Title
        ├── 👁️ View Count
        └── 📋 Copy Button
```

### Command Line Interface

If you prefer using the command line:

1. Run the script:
```bash
python main.py
```

2. Follow the prompts to enter channel URLs
3. Type 'quit' to exit the program

## Web Interface Features

The Streamlit interface provides:

- 📊 Channel Statistics
  - Channel Name
  - Subscriber Count
  - Total Shorts Count

- 🎥 Shorts Analysis
  - Grid view of all shorts
  - Title and view count for each short
  - Copy URL button for easy sharing
  - Direct YouTube Shorts links

- 🎨 User Experience
  - Responsive design
  - Loading indicators
  - Error handling
  - Toast notifications for copy actions
  - Clean card-based layout

## Example Output

### Web Interface
```
Channel Information:
├── Channel Name: MrBeast
└── Subscribers: 240M

Shorts Analysis (25 videos):
├── Card 1
│   ├── Title: I Spent 7 Days Buried Alive!
│   ├── Views: 54M
│   └── [Copy URL] button → https://www.youtube.com/shorts/kuu6nSI74H8
├── Card 2
│   ├── Title: Would You Swim With Sharks For $100,000?
│   ├── Views: 122M
│   └── [Copy URL] button → https://www.youtube.com/shorts/abc123xyz
└── ...
```

### CLI Output
```
Channel Information:
Channel Name: MrBeast
Subscribers: 240M subscribers

Shorts Analysis:
Total Shorts Found: 25

Individual Shorts:
1. Title: I Spent 7 Days Buried Alive!
   Views: 54M
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

5. **Streamlit Interface Not Loading**
   - Check if Streamlit is installed: `pip install streamlit`
   - Verify you're running the correct file: `streamlit run streamlit_app.py`
   - Try clearing your browser cache

6. **Copy Button Not Working**
   - Ensure pyperclip is installed: `pip install pyperclip`
   - On Linux, install xclip: `sudo apt-get install xclip`

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

## New Features

### Card Layout
- Modern, shadow-boxed design
- Clear visual hierarchy
- Responsive grid system
- Visual indicators for views

### Copy Functionality
- One-click copy buttons
- Toast notifications
- Clipboard API integration
- Visual feedback

### Metrics Display
- Clean, modern metrics for channel stats
- Visual separation of data
- Clear data hierarchy
