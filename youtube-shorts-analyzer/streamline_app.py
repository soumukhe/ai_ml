import streamlit as st
import pyperclip
from main import YoutubeShortsAnalyzer
import time

def copy_to_clipboard(text):
    pyperclip.copy(text)

st.set_page_config(
    page_title="📊 YouTube Shorts Analyzer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 YouTube Shorts Analyzer")
st.markdown("""
    Analyze YouTube channels and extract information about their Shorts videos.
    Enter a channel URL below to get started!
""")

# Input for channel URL
channel_url = st.text_input(
    "Enter YouTube Channel URL",
    placeholder="https://www.youtube.com/@ChannelName"
)

if st.button("Analyze Channel", type="primary"):
    if channel_url:
        try:
            with st.spinner("Analyzing channel..."):
                analyzer = YoutubeShortsAnalyzer()
                
                # Create placeholder for channel info
                channel_info = st.empty()
                
                # Create container for shorts
                shorts_container = st.container()
                
                # Analyze channel
                analyzer.analyze_channel(channel_url)
                
                # Display channel information
                with channel_info:
                    st.subheader("Channel Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Channel Name", analyzer.channel_stats["channel_name"])
                    with col2:
                        st.metric("Subscribers", analyzer.channel_stats["subscribers"])
                
                # Display shorts in a grid
                with shorts_container:
                    st.subheader(f"Shorts Analysis ({len(analyzer.shorts_data)} videos found)")
                    
                    # Create columns for the grid
                    cols = st.columns(3)
                    
                    # Display shorts in cards
                    for idx, short in enumerate(analyzer.shorts_data):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            with st.container():
                                st.markdown(
                                    f"""
                                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 5px;'>
                                        <h4>{short['title']}</h4>
                                        <p>Views: {short['views']}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                if st.button(f"Copy Title 📋", key=f"copy_{idx}"):
                                    copy_to_clipboard(short['title'])
                                    st.toast("Title copied to clipboard!")
                                st.markdown("---")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a YouTube channel URL")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        Made with ❤️ using Streamlit | 
        <a href='https://github.com/yourusername/youtube-shorts-analyzer' target='_blank'>GitHub</a>
    </div>
""", unsafe_allow_html=True) 
