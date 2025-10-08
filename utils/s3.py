import boto3
from io import BytesIO
# from PyPDF2 import PdfReader
import os

from config.settings import settings

class S3FileReader:
    def __init__(self):
        self.s3_client = boto3.client('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                      aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                                      )

    def get_s3_file(self, s3_path: str) -> BytesIO:
        """Get file from S3 bucket as bytes"""
        bucket, key = self._parse_s3_path(s3_path)
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        return BytesIO(response['Body'].read())

    def _parse_s3_path(self, s3_path: str) -> tuple:
        """Parse S3 path into bucket and key"""
        path = s3_path.replace('s3://', '')
        bucket = path.split('/')[0]
        key = '/'.join(path.split('/')[1:])
        return bucket, key

    # def read_pdf(self, file_path: str) -> str:
    #     """Read PDF content from S3 or local path"""
    #     if file_path.startswith('s3://'):
    #         file_bytes = self.get_s3_file(file_path)
    #         pdf = PdfReader(file_bytes)
    #     else:
    #         pdf = PdfReader(file_path)
    #     return " ".join(page.extract_text() for page in pdf.pages)

    def read_text(self, file_path: str) -> str:
        """Read text content from S3 or local path"""
        if file_path.startswith('s3://'):
            file_bytes = self.get_s3_file(file_path)
            return file_bytes.getvalue().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
    
    def read_csv_or_excel(self, file_path: str) -> bytes:
        """
        Read CSV or Excel file content from S3 or local path
        """
        # Hardcoded file path removal (you might want to remove this in production)
        import io
        
        # Check if the file is from S3
        if file_path.startswith('s3://'):
            # Get file bytes from S3
            file_bytes = self.get_s3_file(file_path)
            
            if isinstance(file_bytes, io.BytesIO):
                file_bytes = file_bytes.getvalue()
            
            return file_bytes
        
        # Handle local files
        else:
            # Read file as bytes
            with open(file_path, 'rb') as file:
                return file.read()
