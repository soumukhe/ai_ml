import streamlit as st
from pathlib import Path
from user_manager import UserManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize session state variables"""
    if 'user_manager' not in st.session_state:
        st.session_state.user_manager = UserManager()
    if 'file_manager' not in st.session_state:
        st.session_state.file_manager = None
    if 'chroma_manager' not in st.session_state:
        st.session_state.chroma_manager = None
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'sentiment_model' not in st.session_state:
        st.session_state.sentiment_model = None
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False

def login_page():
    """Display login page and handle user authentication"""
    if st.session_state.is_logged_in:
        return True

    st.title("Solutions Domain Analyzer 🔍")
    
    # Create tabs for login and register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        if st.button("Login", key="login_button"):
            if st.session_state.user_manager.user_exists(username):
                st.session_state.current_user = username
                st.session_state.is_logged_in = True
                st.rerun()
            else:
                st.error("User does not exist")
                return False
    
    with tab2:
        new_username = st.text_input("Username", key="register_username")
        if st.button("Register", key="register_button"):
            if st.session_state.user_manager.user_exists(new_username):
                st.error("User already exists")
                return False
            else:
                st.session_state.user_manager.create_user(new_username)
                st.session_state.current_user = new_username
                st.session_state.is_logged_in = True
                st.success("Registration successful!")
                st.rerun()
    
    return st.session_state.is_logged_in

def get_current_user():
    """Get the current logged-in user"""
    return st.session_state.current_user if 'current_user' in st.session_state else None

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Solutions Domain Analyzer",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS for a more professional look
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            height: 3em;
            margin-top: 1em;
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 4px;
        }
        .stTabs {
            margin-top: 2em;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            justify-content: center;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.2rem;
            font-weight: 600;
            color: #495057;
            padding: 0.75rem 2rem;
            background: white;
            border-radius: 4px;
            margin: 0 0.5rem;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #0d6efd;
            background: #e9ecef;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            color: #0d6efd;
            border-bottom: 3px solid #0d6efd;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Run the login page
    login_page() 