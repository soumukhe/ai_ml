# Proposed Enhancements for Solutions Domain Analyzer

## Multi-User Support with Cached Embeddings

### Overview
This enhancement aims to improve performance and add multi-user support by:
1. Caching embeddings in ChromaDB
2. Adding user management
3. Implementing file change detection
4. Creating isolated user spaces

### Technical Implementation

#### 1. User Management System
```python
class UserManager:
    def __init__(self, base_dir="user_data"):
        self.base_dir = base_dir
        
    def create_user(self, username):
        user_dir = os.path.join(self.base_dir, username)
        chroma_dir = os.path.join(user_dir, "chroma_db")
        uploads_dir = os.path.join(user_dir, "uploads")
        
        # Create necessary directories
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)
        os.makedirs(uploads_dir, exist_ok=True)
        
        return user_dir
```

#### 2. File Hash Tracking
```python
def get_file_hash(file_path):
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
```

#### 3. ChromaDB Integration
```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self, user_dir):
        self.user_dir = user_dir
        self.chroma_dir = os.path.join(user_dir, "chroma_db")
        self.hash_file = os.path.join(user_dir, "file_hashes.json")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
    def load_or_create_embeddings(self, file_path, texts):
        """Load existing embeddings or create new ones"""
        current_hash = get_file_hash(file_path)
        stored_hash = self._get_stored_hash(file_path)
        
        if current_hash == stored_hash:
            # Load existing embeddings
            return Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings
            )
        else:
            # Create new embeddings
            vectorstore = Chroma.from_texts(
                texts,
                self.embeddings,
                persist_directory=self.chroma_dir
            )
            self._store_hash(file_path, current_hash)
            return vectorstore
```

#### 4. Streamlit UI for User Management
```python
def user_management_tab():
    st.title("User Management")
    
    # Create new user
    with st.expander("Create New User"):
        new_username = st.text_input("Enter username")
        if st.button("Create User"):
            user_manager = UserManager()
            user_dir = user_manager.create_user(new_username)
            st.success(f"Created user directory: {user_dir}")
    
    # Manage files
    with st.expander("Manage Files"):
        username = st.selectbox("Select User", os.listdir("user_data"))
        if username:
            user_dir = os.path.join("user_data", username)
            uploads_dir = os.path.join(user_dir, "uploads")
            
            # Show existing files
            files = os.listdir(uploads_dir)
            if files:
                for file in files:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(file)
                    with col2:
                        if st.button("Delete", key=f"del_{file}"):
                            os.remove(os.path.join(uploads_dir, file))
                            st.rerun()
```

#### 5. Integration with Main App
```python
def process_domain_data(df, domain, user_dir):
    embedding_manager = EmbeddingManager(user_dir)
    
    # Get texts for embedding
    texts = df['Reason_W_AddDetails'].tolist()
    
    # Load or create embeddings
    vectorstore = embedding_manager.load_or_create_embeddings(
        file_path=current_file_path,
        texts=texts
    )
    
    # Use the embeddings for similarity search
    embeddings = vectorstore.get_embeddings()
    # ... rest of processing
```

### Required Dependencies
Add to requirements.txt:
```txt
chromadb>=0.4.0
langchain-community>=0.0.10
```

### Directory Structure
```
user_data/
  ├── user1/
  │   ├── chroma_db/
  │   ├── uploads/
  │   └── file_hashes.json
  └── user2/
      ├── chroma_db/
      ├── uploads/
      └── file_hashes.json
```

### Key Benefits

1. **Performance Optimization**:
   - Cached embeddings for faster processing
   - Skip re-embedding unchanged files
   - Persistent storage of embeddings

2. **Multi-User Support**:
   - Isolated user data directories
   - Personal file management
   - User-specific embedding caches

3. **File Management**:
   - Upload/delete files through UI
   - Track file changes
   - Manage storage space

4. **Security**:
   - Data isolation between users
   - Controlled access to files
   - Protected embedding storage

### Implementation Steps

1. **Phase 1: Basic Structure**
   - Set up user directory structure
   - Implement user creation
   - Add file upload to user directories

2. **Phase 2: ChromaDB Integration**
   - Implement embedding caching
   - Add file hash tracking
   - Create embedding manager

3. **Phase 3: UI Enhancement**
   - Add user management tab
   - Create file management interface
   - Implement file deletion

4. **Phase 4: Integration**
   - Modify main processing logic
   - Add user session management
   - Implement error handling

### Notes
- Implementation should be done incrementally
- Each phase should be tested thoroughly
- Consider adding user authentication in future
- May need to implement storage quotas
- Consider backup strategy for embeddings 