from dotenv import load_dotenv
import os

from utils.singleton import singleton


@singleton
class Settings:
    def __init__(self):
        load_dotenv()

        # API Keys
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.OPENROUTER_API_URL = os.getenv('OPENROUTER_API_URL')

        # ChromaDB Config
        self.CHROMA_API_KEY = os.getenv('CHROMA_API_KEY')
        self.CHROMA_TENANT = os.getenv('CHROMA_TENANT')
        self.CHROMA_DATABASE = os.getenv('CHROMA_DATABASE')

        # LLM MODEL - Try different free models to avoid rate limits
        self.OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'deepseek/deepseek-r1-0528-qwen3-8b:free')

        # Gemini API for lightweight tasks
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyC3I937nk4tfVXZRkeBI48QZY-qheWUj2g')
        self.DEVELOPMENT_MODEL = os.getenv('DEVELOPMENT_MODEL', 'gemini-2.0-flash')

        
        #AWS s3
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_KEY')
        self.AWS_REGION = os.getenv('AWS_REGION')
        self.S3_BUCKET = os.getenv('S3_BUCKET')

        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')


    def setup_environment(self):
        """Setup environment variables"""
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.HUGGINGFACE_TOKEN
        os.environ['OPENAI_API_KEY'] = self.OPENAI_API_KEY


settings = Settings()
