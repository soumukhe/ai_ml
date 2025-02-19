import os
import json
import hashlib
from pathlib import Path
import shutil
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, user_dir):
        """Initialize FileManager for a specific user.
        
        Args:
            user_dir (Path): User's base directory
        """
        self.user_dir = Path(user_dir)
        self.uploads_dir = self.user_dir / "uploads"
        self.temp_dir = self.user_dir / "temp"
        self.hash_file = self.user_dir / "file_hashes.json"
        self._ensure_dirs()
        self._load_hashes()
        
    def _ensure_dirs(self):
        """Ensure all necessary directories exist"""
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_hashes(self):
        """Load file hashes from JSON file"""
        try:
            if self.hash_file.exists():
                with open(self.hash_file, 'r') as f:
                    self.hashes = json.load(f)
            else:
                self.hashes = {}
        except Exception as e:
            logger.error(f"Error loading hashes: {str(e)}")
            self.hashes = {}
            
    def _save_hashes(self):
        """Save file hashes to JSON file"""
        try:
            with open(self.hash_file, 'w') as f:
                json.dump(self.hashes, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving hashes: {str(e)}")
            
    def get_file_hash(self, file_path):
        """Calculate MD5 hash of file.
        
        Args:
            file_path (Path): Path to file
            
        Returns:
            str: MD5 hash of file
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return None
            
    def save_file(self, file_obj, filename):
        """Save uploaded file and track its hash.
        
        Args:
            file_obj: File-like object to save
            filename (str): Name to save file as
            
        Returns:
            Path: Path to saved file
        """
        try:
            # Save file
            file_path = self.uploads_dir / filename
            with open(file_path, 'wb') as f:
                f.write(file_obj.getbuffer())
                
            # Calculate and store hash
            file_hash = self.get_file_hash(file_path)
            if file_hash:
                self.hashes[str(file_path)] = {
                    'hash': file_hash,
                    'last_modified': str(datetime.now())
                }
                self._save_hashes()
                
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving file {filename}: {str(e)}")
            raise
            
    def file_changed(self, file_path):
        """Check if file has changed since last hash.
        
        Args:
            file_path (Path): Path to file to check
            
        Returns:
            bool: True if file has changed or is new, False otherwise
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return True
            
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.hashes.get(str(file_path), {}).get('hash')
        
        return current_hash != stored_hash
        
    def list_files(self):
        """List all files in uploads directory.
        
        Returns:
            list: List of filenames
        """
        return [f.name for f in self.uploads_dir.iterdir() if f.is_file()]
        
    def delete_file(self, filename):
        """Delete a file and its hash record"""
        try:
            # Get the file path
            file_path = self.uploads_dir / filename
            
            # Delete the original file if it exists
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted file: {file_path}")
            
            # Delete associated processed files
            processed_files = [
                self.user_dir / "processed_data.csv",
                self.user_dir / "processed_data.pkl",
                self.user_dir / f"{filename}_processed.csv",
                self.user_dir / f"{filename}_processed.pkl"
            ]
            
            for proc_file in processed_files:
                if proc_file.exists():
                    proc_file.unlink()
                    logger.info(f"Deleted processed file: {proc_file}")
            
            # Remove hash record
            if self.hash_file.exists():
                with open(self.hash_file, 'r') as f:
                    hashes = json.load(f)
                if filename in hashes:
                    del hashes[filename]
                    with open(self.hash_file, 'w') as f:
                        json.dump(hashes, f, indent=4)
                    logger.info(f"Removed hash record for: {filename}")
            
            # Reset Streamlit session state if available
            try:
                import streamlit as st
                # Reset data-related state
                st.session_state.processed_df = None
                st.session_state.df = None
                st.session_state.uploaded_file = None
                st.session_state.processed_df_columns = None
                st.session_state.selected_domain = None
                
                # Reset UI-related state
                st.session_state.active_tab = 0
                st.session_state.fuzzy_search_df = None
                st.session_state.current_report = None
                
                # Reset tab states
                st.session_state.main_tab = None
                st.session_state.fuzzy_search_tab = None
                st.session_state.reports_tab = None
                st.session_state.file_management_tab = None
                
                # Clear cached data
                st.cache_data.clear()
                st.cache_resource.clear()
                
                logger.info("Reset Streamlit session state")
            except ImportError:
                # Streamlit not available, skip session state reset
                pass
                
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            raise
            
    def get_file_metadata(self, filename):
        """Get metadata for file.
        
        Args:
            filename (str): Name of file
            
        Returns:
            dict: File metadata if exists, None otherwise
        """
        file_path = str(self.uploads_dir / filename)
        return self.hashes.get(file_path) 