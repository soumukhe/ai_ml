import streamlit as st
import time
from main_rag import RAGSystem
import os

# Set page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f3ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
    }
    code {
        padding: 0.2em 0.4em;
        border-radius: 3px;
    }
    pre {
        padding: 1em;
        border-radius: 5px;
        background-color: #f6f8fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        # Load documents when first initializing
        with st.spinner('Loading and embedding documents...'):
            st.session_state.rag_system.load_documents()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling."""
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <div><strong>{'You' if role == 'user' else '🤖 Assistant'}</strong></div>
            <div class="message-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📚 Document Q&A System")
        st.markdown("---")
        st.markdown("""
        ### About
        This system allows you to ask questions about your documents using advanced AI. 
        The system will:
        - Search through your documents
        - Find relevant information
        - Provide detailed answers with source references
        """)
        
        st.markdown("---")
        st.markdown("""
        ### Data Source
        The documents in this system were automatically crawled and processed using **crawl4AI** - 
        an intelligent crawler designed to extract documentation and convert it to markdown format.
        
        All documents are stored in the `data` directory and are automatically processed when changes are detected.
        """)
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Document Status")
        # Add document count
        try:
            doc_count = len([f for f in os.listdir("data") if f.endswith('.md')])
            st.info(f"📂 {doc_count} documents loaded and ready for queries")
        except Exception:
            st.info("Documents are loaded and ready for queries")
    
    # Main chat interface
    st.markdown("### Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message["role"], message["content"])
    
    # Input area
    st.markdown("---")
    with st.container():
        # Use a form to handle input properly
        with st.form(key="question_form", clear_on_submit=True):
            question = st.text_area("Your question:", key="question_input", height=100)
            submit_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
            
            if submit_button and question:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                # Get response
                with st.spinner('Thinking...'):
                    response = st.session_state.rag_system.query(question)
                
                # Add assistant response to chat
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to update chat display
                st.rerun()
        
        st.markdown("*Press Enter to start a new line, Ctrl+Enter to submit*")

if __name__ == "__main__":
    main() 