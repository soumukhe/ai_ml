import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
import base64
import urllib3
import ssl
from fpdf import FPDF
import io

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Configure SSL context for legacy support
class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            ssl_context=ctx
        )

def get_legacy_session():
    session = requests.Session()
    adapter = CustomHttpAdapter()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

# Page configuration
st.set_page_config(
    page_title="Web Scraper with GPT-4o",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .output-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-message {
        color: #28a745;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        color: #dc3545;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .summary-box {
        background-color: #e9ecef;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
        border-left: 4px solid #007bff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #ffffff;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0f2f6;
    }
    .download-section {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_summary_pdf(title, summary, url):
    """Create a PDF with the content summary."""
    try:
        # Create PDF object
        pdf = FPDF()
        pdf.add_page()
        
        # Set up fonts - use built-in Helvetica to avoid font issues
        pdf.set_font('Helvetica', '', 12)
        
        # Title
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "Content Summary", ln=True, align='C')
        pdf.ln(10)
        
        # URL and timestamp
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 10, f"Source: {url}", ln=True)
        pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(10)
        
        # Page title
        pdf.set_font('Helvetica', 'B', 14)
        pdf.multi_cell(0, 10, f"Page Title: {title}")
        pdf.ln(10)
        
        # Summary content
        pdf.set_font('Helvetica', '', 12)
        
        # Split summary into paragraphs and write them
        paragraphs = summary.split('\n')
        for para in paragraphs:
            if para.strip():  # Only write non-empty paragraphs
                pdf.multi_cell(0, 10, para.strip())
                pdf.ln(5)
        
        # Save PDF to bytes buffer
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        return pdf_bytes
        
    except Exception as e:
        st.error(f"PDF Generation Error: {str(e)}")
        return None

def get_download_link_pdf(content, filename):
    """Generate a download link for PDF content."""
    try:
        if content is None:
            return ""
        b64 = base64.b64encode(content).decode()
        html = f'''
            <a href="data:application/pdf;base64,{b64}" 
               download="{filename}"
               style="text-decoration: none;">
                <button style="
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 12px 24px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 4px;">
                    📥 Download Summary PDF
                </button>
            </a>
        '''
        return html
    except Exception as e:
        st.error(f"Download Link Error: {str(e)}")
        return ""

def validate_url(url):
    """Validate if the given URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def get_content_summary(content):
    """Generate a summary of the content using GPT-4o."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("Error: OPENAI_API_KEY not found in .env file")
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
            
        client = OpenAI(**client_kwargs)
        
        with st.spinner("Generating content summary..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates clear, concise summaries of web content. Focus on the main points and key takeaways."},
                    {"role": "user", "content": f"""Please provide a clear and concise summary of the following content. 
                    Focus on the main points, key ideas, and important takeaways. Make it easy to understand:

                    {content}"""}
                ],
                temperature=0.3,
                max_tokens=1000
            )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating summary: {str(e)}")
        return None

def convert_to_markdown_with_gpt4(content):
    """Convert the scraped content to markdown using GPT-4o."""
    try:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            st.error("Error: OPENAI_API_KEY not found in .env file")
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
            
        client = OpenAI(**client_kwargs)
        
        with st.spinner("Converting content to markdown with GPT-4o..."):
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that converts web content to clean markdown format."},
                    {"role": "user", "content": f"""Please convert the following web content into well-formatted markdown. 
                    Make it clean, organized, and easy to read. Preserve all important information while improving readability:

                    {content}"""}
                ],
                temperature=0.3,
                max_tokens=4096
            )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in markdown conversion: {str(e)}")
        return None

def scrape_webpage(url):
    """Scrape the webpage and extract relevant information."""
    try:
        # Get timeout from .env or use default
        timeout = int(os.getenv('REQUESTS_TIMEOUT', 30))
        
        with st.spinner("Fetching webpage content..."):
            # Send request with a common user agent and legacy SSL support
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Use legacy SSL session
            session = get_legacy_session()
            response = session.get(url, headers=headers, timeout=timeout)
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
        
        return raw_content
        
    except requests.RequestException as e:
        st.error(f"Error fetching the webpage: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def get_download_link(content, filename):
    """Generate a download link for the markdown content."""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:text/markdown;base64,{b64}" download="{filename}" class="download-button">Download Markdown File</a>'

def main():
    # Header
    st.title("🌐 Web Scraper with GPT-4o")
    st.markdown("Convert any webpage into well-formatted markdown using GPT-4o")
    
    # URL Input
    url = st.text_input("Enter the webpage URL:", placeholder="https://example.com")
    
    if url:
        if not validate_url(url):
            st.error("Invalid URL format. Please enter a valid URL (e.g., https://www.example.com)")
        else:
            # Scrape and process the webpage
            raw_content = scrape_webpage(url)
            
            if raw_content:
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["📝 Raw Content", "📋 Summary", "📄 Markdown"])
                
                # Tab 1: Raw Content
                with tab1:
                    st.text_area("Raw Scraped Content", raw_content, height=400)
                
                # Tab 2: Summary
                summary_content = None
                with tab2:
                    summary_content = get_content_summary(raw_content)
                    if summary_content:
                        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                        st.markdown(summary_content)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Tab 3: Markdown
                with tab3:
                    markdown_content = convert_to_markdown_with_gpt4(raw_content)
                    if markdown_content:
                        st.markdown(markdown_content)
                        
                        # Add download button below the markdown content
                        st.markdown("---")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"scraped_content_{timestamp}.md"
                        st.markdown(get_download_link(markdown_content, filename), unsafe_allow_html=True)
                
                # Add download section for both formats
                if summary_content and markdown_content:
                    st.markdown("---")
                    st.markdown('<div class="download-section">', unsafe_allow_html=True)
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create and offer PDF download
                        try:
                            pdf_content = create_summary_pdf(
                                raw_content.split('\n')[1].strip(),  # Get the title
                                summary_content,
                                url
                            )
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            pdf_filename = f"summary_{timestamp}.pdf"
                            st.markdown(get_download_link_pdf(pdf_content, pdf_filename), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error creating PDF: {str(e)}")
                    
                    with col2:
                        # Markdown download
                        md_filename = f"content_{timestamp}.md"
                        st.markdown(get_download_link(markdown_content, md_filename), unsafe_allow_html=True)
                
                # Add success message
                st.markdown('<div class="success-message">✅ Content successfully processed!</div>', 
                          unsafe_allow_html=True)

if __name__ == "__main__":
    main() 