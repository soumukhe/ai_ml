# Web Scraper with GPT-4o

A web application that extracts content from web pages, uses GPT-4o to generate concise summaries and convert content into well-formatted markdown, with options to download both PDF summaries and markdown files.

## Download

### Method 1: Full Repository Clone
```bash
git clone https://github.com/soumukhe/ai_ml.git
cd ai_ml/WebScraperWithGPT-4o
```

### Method 2: Sparse Checkout (Recommended)
If you only want this specific project without downloading the entire repository:

```bash
# Create and enter a new directory
mkdir my_demo && cd my_demo

# Initialize git
git init

# Add the remote repository
git remote add -f origin https://github.com/soumukhe/ai_ml.git

# Enable sparse checkout
git config core.sparseCheckout true

# Specify the subdirectory you want to clone
echo 'WebScraperWithGPT-4o' >> .git/info/sparse-checkout

# Pull the subdirectory
git pull origin master

# Enter the project directory
cd WebScraperWithGPT-4o
```

## Features

- Modern Streamlit web interface with tabbed layout
- Real-time web scraping with SSL support
- GPT-4o powered content processing:
  - Smart content summarization
  - Markdown conversion
- Multiple download options:
  - PDF summary with professional formatting
  - Full markdown content
- Legacy SSL support for corporate websites
- Progress indicators and error handling
- Responsive design
- Environment variable based configuration

## Setup

1. Ensure you have Python 3.x installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   - Copy `.env.template` to `.env`:
     ```
     cp .env.template .env
     ```
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
   - Optional: Configure additional settings in `.env` if needed

## Usage

### Method 1: Using the Process Manager (Recommended)
```bash
python run.py
```
This will:
- Start the Streamlit server in a subprocess
- Automatically open your default browser
- Handle server cleanup on exit (Ctrl+C)
- Find an available port automatically

### Method 2: Direct Streamlit Launch
```bash
streamlit run app.py
```

## Application Flow

1. Enter a URL in the input field and press Enter

2. The app will process the content and show three tabs:
   - 📝 Raw Content: Shows the scraped content
   - 📋 Summary: Displays a concise GPT-4o generated summary
   - 📄 Markdown: Shows the converted markdown

3. Download Options:
   - Download Summary PDF: Get a professionally formatted PDF
   - Download Markdown: Get the full content in markdown format

## Features in Detail

- **URL Input**: Enter any valid webpage URL
- **Raw Content View**: See exactly what was scraped from the webpage
- **Smart Summary**: GPT-4o generated concise summary with key points
- **Markdown Preview**: See the GPT-4o converted markdown in real-time
- **PDF Generation**: Professional PDF with:
  - Title page
  - Source URL
  - Timestamp
  - Formatted content
  - Blue accent styling
- **Download Options**: Multiple formats for flexibility
- **Error Handling**: Clear error messages for troubleshooting
- **Progress Indicators**: Visual feedback during processing
- **Responsive Layout**: Works well on different screen sizes
- **SSL Support**: Handles corporate websites with legacy SSL

## Process Management

- If running with `run.py`: Press Ctrl+C in the terminal
- If running directly with Streamlit: Close the terminal or use Ctrl+C
- Alternative commands:
  ```bash
  # View all Streamlit processes
  ps -aef | grep streamlit
  
  # Kill all Streamlit processes
  pkill -f streamlit
  ```