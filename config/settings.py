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
        self.OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'mistralai/mistral-7b-instruct:free')

        # Gemini API for lightweight tasks
        self.GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'AIzaSyBvp8WKlFMu_oB-NGmskIdVctdVhgkkLU4')
        self.DEVELOPMENT_MODEL = os.getenv('DEVELOPMENT_MODEL', 'gemini-2.0-flash')

        # XAI API Key
        self.XAI_API_KEY = os.getenv('XAI_API_KEY')
        # Deepseek API Key
        self.DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


        
        #AWS s3
        self.AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_ID')
        self.AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_KEY')
        self.AWS_REGION = os.getenv('AWS_REGION')
        self.S3_BUCKET = os.getenv('S3_BUCKET')

        self.UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')

        # Hugging Face Embeddings and Reranker Config
        self.HF_API_KEY = os.getenv('HF_API_KEY')
        self.HF_EMBED_MODEL = os.getenv('HF_EMBED_MODEL', 'BAAI/bge-m3')
        self.HF_RERANKER_MODEL = os.getenv('HF_RERANKER_MODEL', 'cross-encoder/ms-marco-TinyBERT-L-2-v2')
        self.EMBED_BATCH_SIZE = int(os.getenv('EMBED_BATCH_SIZE', 64))
        self.TOP_K = int(os.getenv('TOP_K', 50))
        self.RERANK_N = int(os.getenv('RERANK_N', 20))
        self.MAX_RERANK_DOCS = int(os.getenv('MAX_RERANK_DOCS', 15))  # Limit docs for reranking speed


    def setup_environment(self):
        """Setup environment variables"""
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = self.HUGGINGFACE_TOKEN
        os.environ['OPENAI_API_KEY'] = self.OPENAI_API_KEY


settings = Settings()
