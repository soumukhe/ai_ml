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

class UserManager:
    def __init__(self, base_dir="user_data"):
        """Initialize UserManager with base directory for user data.
        
        Args:
            base_dir (str): Base directory for all user data
        """
        self.base_dir = Path(base_dir)
        self._ensure_base_dir()
        
    def _ensure_base_dir(self):
        """Ensure base directory exists"""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_user(self, username):
        """Create a new user directory structure.
        
        Args:
            username (str): Username to create directories for
            
        Returns:
            Path: Path to user directory
        """
        try:
            # Create user directory path
            user_dir = self.base_dir / username
            
            # Create necessary subdirectories
            directories = {
                'chroma_db': user_dir / "chroma_db",
                'uploads': user_dir / "uploads",
                'temp': user_dir / "temp",
                'reports': user_dir / "reports"
            }
            
            # Create all directories
            for dir_path in directories.values():
                dir_path.mkdir(parents=True, exist_ok=True)
                
            # Create user metadata file
            metadata = {
                'username': username,
                'created_at': str(datetime.now()),
                'last_access': str(datetime.now())
            }
            
            with open(user_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Created user directory structure for {username}")
            return user_dir
            
        except Exception as e:
            logger.error(f"Error creating user directory for {username}: {str(e)}")
            raise
            
    def get_user_dir(self, username):
        """Get user directory path if it exists.
        
        Args:
            username (str): Username to get directory for
            
        Returns:
            Path: Path to user directory if exists, None otherwise
        """
        user_dir = self.base_dir / username
        return user_dir if user_dir.exists() else None
        
    def user_exists(self, username):
        """Check if user exists.
        
        Args:
            username (str): Username to check
            
        Returns:
            bool: True if user exists, False otherwise
        """
        return (self.base_dir / username).exists()
        
    def delete_user(self, username):
        """Delete a user's directory structure.
        
        Args:
            username (str): Username to delete
        """
        try:
            user_dir = self.base_dir / username
            if user_dir.exists():
                shutil.rmtree(user_dir)
                logger.info(f"Deleted user directory for {username}")
            else:
                logger.warning(f"User directory for {username} does not exist")
        except Exception as e:
            logger.error(f"Error deleting user directory for {username}: {str(e)}")
            raise
            
    def list_users(self):
        """List all users.
        
        Returns:
            list: List of usernames
        """
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        
    def get_user_metadata(self, username):
        """Get user metadata.
        
        Args:
            username (str): Username to get metadata for
            
        Returns:
            dict: User metadata if exists, None otherwise
        """
        try:
            metadata_file = self.base_dir / username / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error reading metadata for {username}: {str(e)}")
            return None
            
    def update_last_access(self, username):
        """Update user's last access time.
        
        Args:
            username (str): Username to update
        """
        try:
            metadata_file = self.base_dir / username / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                metadata['last_access'] = str(datetime.now())
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating last access for {username}: {str(e)}") 