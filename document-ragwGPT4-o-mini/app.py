import os
import streamlit as st
from rag_system import RAGSystem
import humanize
from datetime import datetime
from typing import List
import shutil
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Documentation RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("📚 Documentation RAG")
st.markdown("""
A Retrieval-Augmented Generation system for intelligent document Q&A.
""")

# Footer
st.markdown("""
---
Created with ❤️ by Soumitra Mukherji 🚀
""", unsafe_allow_html=True)

# Increase maximum upload size to 1GB
st.config.set_option('server.maxUploadSize', 1024)  # Size in MB

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.user_id = "default"
    st.session_state.theme = "light"
    st.session_state.rag_system = RAGSystem(user_id="default")
    st.session_state.initialized = True

# Get users once and reuse
users = RAGSystem.get_all_users()
user_ids = [user["user_id"] for user in users]

# Ensure user_id is valid
if st.session_state.user_id not in user_ids:
    st.session_state.user_id = "default"
    st.session_state.rag_system = RAGSystem(user_id="default")

def switch_user(new_user_id: str):
    """Switch to a different user."""
    if new_user_id != st.session_state.user_id:
        st.session_state.user_id = new_user_id
        st.session_state.rag_system = RAGSystem(user_id=new_user_id)
        st.rerun()

# Theme toggle
def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

# Custom CSS for themes
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stButton button { background-color: #2E2E2E; color: #FFFFFF; }
        .stTextInput input { background-color: #2E2E2E; color: #FFFFFF; }
        </style>
    """, unsafe_allow_html=True)

# Display current user and debug info
st.title("📚 Cisco Documentation RAG")

# Theme toggle in the header
col1, col2, col3 = st.columns([6, 1, 1])
with col2:
    if st.button("🌓 Theme"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

# Apply theme
if st.session_state.theme == "dark":
    st.markdown("""
        <style>
        .stApp { background-color: #1E1E1E; color: #FFFFFF; }
        .stButton button { background-color: #2E2E2E; color: #FFFFFF; }
        .stTextInput input { background-color: #2E2E2E; color: #FFFFFF; }
        .stTextArea textarea { background-color: #2E2E2E; color: #FFFFFF; }
        .stSelectbox select { background-color: #2E2E2E; color: #FFFFFF; }
        </style>
    """, unsafe_allow_html=True)

st.caption(f"Current User: {st.session_state.user_id}")

# Debug info
with st.expander("🔧 Debug Info", expanded=False):
    st.write("Current User ID:", st.session_state.user_id)
    st.write("Session State Keys:", list(st.session_state.keys()))

# Rest of the UI
st.markdown("""
This system allows you to:
* Upload PDF documents
* Ask questions about their content
* Get AI-powered answers with source references
* Save favorite Q&A pairs
* View chat history
""")

# Add security information in a card-like container
st.markdown("### 🔒 Security & Privacy")
security_info = st.container()
with security_info:
    st.info("""
    **Important Security Information:**
    * All document processing is done locally on the machine running this app
    * Document embeddings are generated locally using HuggingFace's all-MiniLM-L6-v2 model
    * No document content is ever sent to external servers
    * Only your final questions are sent to OpenAI's secure API endpoints
    * All embeddings and vector stores are saved securely in the local workspace
    """)

# Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["👤 User", "📁 Documents", "❓ Ask Questions", "⭐ Favorites", "📜 History"])

with tab1:
    # User Management Section
    st.markdown("## User Management")
    
    # Switch/Create User
    users = RAGSystem.get_all_users()
    user_ids = [user["user_id"] for user in users]
    
    # User Switching Section
    st.markdown("### Switch User")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Debug info for user directories
        with st.expander("🔍 Debug Info", expanded=False):
            base_dir = "user_data"
            st.write("User Directories:")
            if os.path.exists(base_dir):
                for user_id in os.listdir(base_dir):
                    user_dir = os.path.join(base_dir, user_id)
                    st.write(f"User: {user_id}")
                    st.write(f"- Directory exists: {os.path.isdir(user_dir)}")
                    st.write(f"- Profile exists: {os.path.exists(os.path.join(user_dir, 'profile.json'))}")
                    st.write(f"- Docs exists: {os.path.exists(os.path.join(user_dir, 'docs'))}")
        
        # Direct user selection
        selected_user = st.radio(
            "Select User",
            user_ids,
            index=user_ids.index(st.session_state.user_id),
            horizontal=True
        )
    
    with col2:
        if st.button("🔄 Switch", use_container_width=True):
            switch_user(selected_user)
        
        # Enhanced cleanup button
        if st.button("🧹 Force Cleanup", use_container_width=True, help="Force remove invalid user data"):
            base_dir = "user_data"
            cleaned = False
            if os.path.exists(base_dir):
                for user_id in os.listdir(base_dir):
                    if user_id == "default":
                        continue  # Skip default user
                        
                    user_dir = os.path.join(base_dir, user_id)
                    try:
                        # Force remove the user directory
                        if os.path.exists(user_dir):
                            shutil.rmtree(user_dir)
                            st.success(f"Removed user directory: {user_id}")
                            cleaned = True
                    except Exception as e:
                        st.error(f"Error removing {user_id}: {str(e)}")
                        # Try alternative removal method
                        try:
                            os.system(f"rm -rf {user_dir}")
                            st.success(f"Forced removal of user directory: {user_id}")
                            cleaned = True
                        except Exception as e2:
                            st.error(f"Force removal also failed for {user_id}: {str(e2)}")
            
            if cleaned:
                st.success("Cleanup completed. Refreshing...")
                time.sleep(1)  # Give time for the filesystem to update
                st.rerun()
            else:
                st.info("No users to clean up")
    
    # Create New User Section
    st.markdown("### Create New User")
    with st.form("create_user_form"):
        new_user_id = st.text_input("New User ID")
        new_user_desc = st.text_area("Description")
        new_user_tags = st.multiselect(
            "Tags",
            ["Network", "Security", "Cloud", "AI", "Documentation", "Project", "Personal", "Other"]
        )
        
        submit_button = st.form_submit_button("Create User")
        if submit_button:
            if new_user_id and new_user_id not in user_ids:
                try:
                    # Create new user
                    new_system = RAGSystem(user_id=new_user_id)
                    # Ensure profile is created
                    new_system.update_profile(
                        description=new_user_desc if new_user_desc else "",
                        tags=new_user_tags if new_user_tags else []
                    )
                    # Switch to new user and force page refresh
                    st.session_state.user_id = new_user_id
                    st.session_state.rag_system = new_system
                    st.success(f"User {new_user_id} created successfully!")
                    time.sleep(1)  # Give time for the filesystem to update
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating user: {str(e)}")
            else:
                if not new_user_id:
                    st.error("Please enter a user ID")
                else:
                    st.error("This user ID already exists. Please choose a different one.")

    # Current User Profile
    if st.session_state.rag_system:
        st.markdown("### Current User Profile")
        stats = st.session_state.rag_system.get_workspace_stats()
        
        # Display stats in columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", stats["total_documents"])
            st.metric("Total Pages", stats["total_pages"])
        with col2:
            st.metric("Total Size", humanize.naturalsize(stats["total_size"]))
            st.metric("Queries Made", stats["total_queries"])
        with col3:
            st.metric("Favorites", stats["total_favorites"])
            st.metric("Last Active", humanize.naturaltime(datetime.fromisoformat(stats["last_active"])))

        # Profile Update Form
        with st.form("update_profile_form"):
            st.markdown("#### Update Profile")
            description = st.text_area(
                "Description",
                value=st.session_state.rag_system.profile.get("description", "")
            )
            tags = st.multiselect(
                "Tags",
                ["Network", "Security", "Cloud", "AI", "Documentation", "Project", "Personal", "Other"],
                default=st.session_state.rag_system.profile.get("tags", [])
            )
            
            if st.form_submit_button("Update Profile"):
                st.session_state.rag_system.update_profile(description, tags)
                st.success("Profile updated successfully!")

    # Delete User Section (moved outside the if block)
    st.markdown("### ⚠️ Danger Zone")
    if st.session_state.user_id != "default":  # Prevent deletion of default user
        with st.expander("Delete Current User", expanded=False):
            st.warning(
                "⚠️ This action will permanently delete the current user and all associated data:\n"
                "* All uploaded documents\n"
                "* Document embeddings\n"
                "* Chat history\n"
                "* Favorites\n"
                "* User profile"
            )
            
            # Require user to type the user ID to confirm deletion
            delete_confirmation = st.text_input(
                f"Type '{st.session_state.user_id}' to confirm deletion:",
                help="This action cannot be undone",
                key="delete_confirmation"
            )
            
            col1, col2 = st.columns([3,1])
            with col1:
                if st.button(
                    "🗑️ Delete User", 
                    type="primary",
                    disabled=delete_confirmation != st.session_state.user_id,
                    help="This will permanently delete all user data",
                    key="delete_user_btn"
                ):
                    if delete_confirmation == st.session_state.user_id:
                        try:
                            # Delete user data
                            st.session_state.rag_system.delete_user_data()
                            # Switch to default user
                            st.session_state.user_id = "default"
                            st.session_state.rag_system = RAGSystem(user_id="default")
                            st.success("User deleted successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting user: {str(e)}")
            with col2:
                if delete_confirmation != st.session_state.user_id and delete_confirmation:
                    st.error("ID doesn't match")
    else:
        st.info("The default user cannot be deleted.")

    # Document Copy Tool
    st.markdown("### 📋 Copy Documents")
    with st.expander("Copy Documents Between Users", expanded=False):
        source_docs = [f for f in os.listdir(st.session_state.rag_system.docs_dir) if f.endswith('.pdf')]
        if source_docs:
            docs_to_copy = st.multiselect(
                "Select Documents to Copy",
                source_docs,
                key="docs_to_copy"
            )
            target_user = st.selectbox(
                "Select Target User",
                [u for u in user_ids if u != st.session_state.user_id],
                key="target_user"
            )
            
            if st.button("📋 Copy Documents", key="copy_docs_btn") and docs_to_copy and target_user:
                success = []
                for doc in docs_to_copy:
                    if st.session_state.rag_system.copy_document(doc, target_user):
                        success.append(doc)
                if success:
                    st.success(f"✅ Copied {len(success)} documents to {target_user}")
                else:
                    st.error("Failed to copy documents")
        else:
            st.info("No documents available to copy")

    # Theme toggle in a small column
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if st.button("🌓 Toggle Theme"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()
    
    # Apply theme
    if st.session_state.theme == "dark":
        st.markdown("""
            <style>
            .stApp { background-color: #1E1E1E; color: #FFFFFF; }
            .stButton button { background-color: #2E2E2E; color: #FFFFFF; }
            .stTextInput input { background-color: #2E2E2E; color: #FFFFFF; }
            </style>
        """, unsafe_allow_html=True)
    
    # Clear History button in History tab
    if st.button("🗑️ Clear History", type="primary"):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear_history()
            st.success("Chat history cleared successfully!")
            st.rerun()

with tab2:
    # Document Management Section
    st.markdown(f"## 📁 Document Management (User: {st.session_state.user_id})")

    # Reprocess button
    if st.button("🔄 Reprocess All Documents", help="Force reprocessing of all documents"):
        with st.spinner("Processing documents..."):
            try:
                num_chunks = st.session_state.rag_system.process_pdfs(force_reload=True)
                st.success(f"✅ Processed {num_chunks} chunks from documents")
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")

    # File uploader
    st.markdown("### Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF files to be processed by the system",
        key=f"pdf_uploader_{st.session_state.user_id}"
    )

    # Process uploaded files
    if uploaded_files:
        processing_msg = st.info("Processing uploaded documents...")
        try:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join("user_data", st.session_state.user_id, "docs", uploaded_file.name)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # Add document metadata with tags
                tags = st.multiselect(
                    f"Add tags for {uploaded_file.name}:",
                    ["Network", "Security", "Cloud", "AI", "Other"],
                    key=f"tags_{uploaded_file.name}"
                )
                st.session_state.rag_system.add_document(file_path, tags)
            
            # Process all documents
            num_chunks = st.session_state.rag_system.process_pdfs()
            processing_msg.success(f"✅ Processed {num_chunks} chunks from documents")
        except Exception as e:
            processing_msg.error(f"❌ Error processing documents: {str(e)}")

    # Display current documents
    st.markdown("### Current Documents")
    try:
        docs_dir = os.path.join("user_data", st.session_state.user_id, "docs")
        if not os.path.exists(docs_dir):
            os.makedirs(docs_dir)
        
        pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.info("No documents uploaded yet.")
        else:
            for pdf_file in pdf_files:
                with st.expander(f"📄 {pdf_file}"):
                    # Get and display metadata
                    metadata = st.session_state.rag_system.get_document_metadata(pdf_file)
                    if metadata:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Pages", metadata.get('pages', 'N/A'))
                        with col2:
                            st.metric("Size", humanize.naturalsize(metadata.get('size', 0)))
                        with col3:
                            upload_date = datetime.fromisoformat(metadata.get('upload_date', ''))
                            st.metric("Uploaded", upload_date.strftime("%Y-%m-%d %H:%M"))
                        
                        if 'tags' in metadata:
                            st.markdown("**Tags:** " + ", ".join(f"`{tag}`" for tag in metadata['tags']))
                    
                    # Delete button
                    if st.button("🗑️ Delete", key=f"delete_{pdf_file}"):
                        try:
                            os.remove(os.path.join(docs_dir, pdf_file))
                            st.session_state.rag_system.process_pdfs()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting {pdf_file}: {str(e)}")
    except Exception as e:
        st.error(f"Error listing documents: {str(e)}")

with tab3:
    # Question and Answer Section
    st.markdown(f"## ☁️ Ask Questions (User: {st.session_state.user_id})")
    question = st.text_input(
        "Enter your question about the documents:",
        help="Ask any question about the content of the uploaded documents"
    )

    col1, col2 = st.columns([6,1])
    with col1:
        ask_button = st.button("🔍 Get Answer", disabled=not question)
    with col2:
        if st.button("🧹 Clear History", key="clear_history_qa"):
            st.session_state.rag_system.clear_history()
            st.rerun()

    if ask_button:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Get response
                    response = st.session_state.rag_system.query(question)
                    
                    # Display answer
                    answer_container = st.container()
                    with answer_container:
                        st.markdown("### Answer")
                        st.markdown(response["answer"])
                        
                        # Add to favorites button
                        if st.button("⭐ Add to Favorites"):
                            sources = [
                                os.path.basename(source.metadata["source"])
                                for source in response["sources"]
                                if hasattr(source, "metadata")
                            ]
                            st.session_state.rag_system.add_to_favorites(
                                question, response["answer"], sources
                            )
                            st.success("Added to favorites!")
                    
                    # Display sources
                    if response["sources"]:
                        st.markdown("### Sources")
                        
                        # Group sources by document
                        doc_sources = {}
                        for source in response["sources"]:
                            doc_name = source.metadata.get('source', 'Unknown Source')
                            if doc_name not in doc_sources:
                                doc_sources[doc_name] = []
                            doc_sources[doc_name].append(source)
                        
                        # Display sources grouped by document
                        for doc_name in sorted(doc_sources.keys()):
                            st.markdown(f"#### 📄 From {os.path.basename(doc_name)}:")
                            for idx, source in enumerate(doc_sources[doc_name], 1):
                                with st.expander(f"Source {idx}"):
                                    st.markdown(source.page_content)
                                    if 'page' in source.metadata:
                                        st.caption(f"Page: {source.metadata['page']}")
                    else:
                        st.info("No specific sources found for this answer.")
                except Exception as e:
                    st.error(f"❌ Error processing your question: {str(e)}")

with tab4:
    # Favorites Section
    st.markdown(f"## ⭐ Favorite Q&A Pairs (User: {st.session_state.user_id})")
    favorites = st.session_state.rag_system.get_favorites()
    
    if not favorites:
        st.info("No favorites saved yet. Click the star button when viewing an answer to save it!")
    else:
        for idx, favorite in enumerate(favorites):
            with st.expander(f"Q: {favorite['question']}", expanded=False):
                st.markdown("**Answer:**")
                st.markdown(favorite['answer'])
                st.markdown("**Sources:**")
                for source in favorite['sources']:
                    st.markdown(f"- {source}")
                st.caption(f"Saved on: {datetime.fromisoformat(favorite['timestamp']).strftime('%Y-%m-%d %H:%M')}")
                if st.button("🗑️ Remove", key=f"remove_favorite_{idx}"):
                    st.session_state.rag_system.remove_from_favorites(idx)
                    st.rerun()

with tab5:
    # History Section
    st.markdown(f"## 📜 Chat History (User: {st.session_state.user_id})")
    
    # Clear History button at the top of History tab
    if st.button("🗑️ Clear History", type="primary", key="clear_history_tab"):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear_history()
            st.success("Chat history cleared successfully!")
            st.rerun()
    
    history = st.session_state.rag_system.get_history()
    
    if not history:
        st.info("No chat history yet. Start asking questions to build your history!")
    else:
        for idx, item in enumerate(reversed(history)):  # Show most recent first
            with st.expander(f"Q: {item['question']}", expanded=False):
                st.markdown("**Answer:**")
                st.markdown(item['answer'])
                st.markdown("**Sources:**")
                for source in item['sources']:
                    st.markdown(f"- {source}")
                st.caption(f"Asked on: {datetime.fromisoformat(item['timestamp']).strftime('%Y-%m-%d %H:%M')}")