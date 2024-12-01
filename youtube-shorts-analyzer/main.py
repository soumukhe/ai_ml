import streamlit as st
import os
import sys
import socket
import webbrowser
import time
from main import YoutubeShortsAnalyzer
import subprocess

def find_available_port(start_port=8501, max_port=8599):
    """Find an available port starting from start_port"""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def run_streamlit_server():
    """Run the Streamlit server"""
    port = find_available_port()
    if not port:
        print("No available ports found")
        sys.exit(1)
    
    # Start Streamlit in a subprocess
    cmd = f"streamlit run {__file__} --server.port {port} --server.headless true -- server"
    subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Open browser
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    
    print(f"\nStreamlit server started at {url}")
    print("To stop the server, use: pkill -f streamlit")
    
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="📊 YouTube Shorts Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 YouTube Shorts Analyzer")
    st.markdown("""
        Analyze YouTube channels and extract information about their Shorts videos.
        Enter a channel URL below to get started!
    """)
    
    # Initialize the analyzer
    analyzer = YoutubeShortsAnalyzer()
    
    # Create input field for channel URL
    channel_url = st.text_input(
        "Enter YouTube Channel URL:",
        placeholder="https://www.youtube.com/@ChannelName"
    )
    
    if st.button("Analyze Channel", type="primary"):
        if channel_url:
            try:
                with st.spinner("Analyzing channel..."):
                    analyzer.analyze_channel(channel_url)
                    
                    # Display channel information in metrics
                    st.header("Channel Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Channel Name", analyzer.channel_stats['channel_name'])
                    with col2:
                        st.metric("Subscribers", analyzer.channel_stats['subscribers'])
                    
                    # Display shorts in a grid layout
                    st.header(f"Shorts Analysis ({len(analyzer.shorts_data)} videos)")
                    
                    # Create columns for the grid
                    cols = st.columns(3)
                    
                    # Display shorts in cards
                    for idx, short in enumerate(analyzer.shorts_data):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            with st.container():
                                # Card styling
                                st.markdown(
                                    f"""
                                    <div style='
                                        padding: 1rem;
                                        border-radius: 0.5rem;
                                        background: #f0f2f6;
                                        margin: 0.5rem 0;
                                        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
                                    '>
                                        <h4 style='margin: 0 0 0.5rem 0;'>{short['title']}</h4>
                                        <p style='
                                            color: #666;
                                            font-size: 0.9rem;
                                            margin: 0.5rem 0;
                                        '>👁️ {short['views']} views</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # Copy button below each card
                                if st.button(f"📋 Copy Title", key=f"copy_{idx}"):
                                    st.toast(f"Copied: {short['title']}")
                                    # Using JavaScript to copy to clipboard
                                    st.markdown(
                                        f"""
                                        <script>
                                            navigator.clipboard.writeText("{short['title']}");
                                        </script>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                st.markdown("<br>", unsafe_allow_html=True)
                                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a YouTube channel URL")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            Made with ❤️ using Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Start the server in a subprocess
        run_streamlit_server()
    elif sys.argv[-1] == "server":
        # Run the Streamlit app
        main()
