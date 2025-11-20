import aiofiles
import shutil
from pathlib import Path
from typing import Optional
from fastapi import UploadFile
import logging

from app.settings import settings
from app.utils.helpers import generate_filename, ensure_directory

logger = logging.getLogger(__name__)


class StorageService:
    """Service for managing file storage."""
    
    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.processed_dir = Path(settings.PROCESSED_DIR)
        self.temp_dir = Path(settings.TEMP_DIR)
        
        # Ensure directories exist
        ensure_directory(self.upload_dir)
        ensure_directory(self.processed_dir)
        ensure_directory(self.temp_dir)
    
    async def save_upload_file(
        self,
        file: UploadFile,
        subfolder: str = "",
        custom_filename: Optional[str] = None
    ) -> str:
        """
        Save an uploaded file to storage.
        
        Args:
            file: FastAPI UploadFile object
            subfolder: Optional subfolder within uploads
            custom_filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        # Determine filename
        if custom_filename:
            filename = custom_filename
        else:
            extension = Path(file.filename).suffix
            filename = generate_filename(prefix="upload", extension=extension)
        
        # Determine save path
        if subfolder:
            save_dir = self.upload_dir / subfolder
            ensure_directory(save_dir)
        else:
            save_dir = self.upload_dir
        
        file_path = save_dir / filename
        
        # Save file
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            raise
    
    async def save_processed_file(
        self,
        content: bytes,
        filename: str,
        subfolder: str = ""
    ) -> str:
        """
        Save processed file content.
        
        Args:
            content: File content as bytes
            filename: Filename
            subfolder: Optional subfolder
            
        Returns:
            Path to saved file
        """
        if subfolder:
            save_dir = self.processed_dir / subfolder
            ensure_directory(save_dir)
        else:
            save_dir = self.processed_dir
        
        file_path = save_dir / filename
        
        try:
            async with aiofiles.open(file_path, 'wb') as out_file:
                await out_file.write(content)
            
            logger.info(f"Processed file saved: {file_path}")
            return str(file_path.relative_to("storage"))

            
        except Exception as e:
            logger.error(f"Error saving processed file: {e}")
            raise
    
    def get_file_url(self, file_path: str) -> str:
        """
        Get URL for accessing a stored file.
        
        Args:
            file_path: Absolute or relative file path
            
        Returns:
            URL path for file access
        """
        # Convert to relative path from storage root
        path = Path(file_path)
    
        storage_root = Path("storage")  # thư mục gốc storage
        
        try:
            rel_path = path.relative_to(storage_root)
        except ValueError:
            # Nếu không thuộc storage, dùng path như hiện tại
            rel_path = path
        
        return f"/storage/{rel_path.as_posix()}"
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if deleted successfully
        """
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    def delete_directory(self, directory_path: str) -> bool:
        """
        Delete a directory and all its contents.
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if deleted successfully
        """
        try:
            path = Path(directory_path)
            if path.exists() and path.is_dir():
                shutil.rmtree(path)
                logger.info(f"Directory deleted: {directory_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting directory: {e}")
            return False
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up temporary files older than specified hours.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of files deleted
        """
        from app.utils.helpers import clean_old_files
        
        deleted = clean_old_files(str(self.temp_dir), max_age_hours)
        logger.info(f"Cleaned up {deleted} temporary files")
        return deleted
