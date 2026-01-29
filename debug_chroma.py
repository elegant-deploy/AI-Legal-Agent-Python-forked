import chromadb
from chromadb.utils import embedding_functions
from config.settings import settings

def check_collections():
    print("running check...")
    try:
        client = chromadb.CloudClient(
            api_key=settings.CHROMA_API_KEY,
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE
        )
        
        collections = client.list_collections()
        print(f"Found {len(collections)} collections.")
        
        for col in collections:
            print(f"\nCollection: {col.name}")
            print(f"Metadata: {col.metadata}")
            
            # Try to get with default (no arg)
            try:
                c = client.get_collection(col.name)
                print(" - Successfully loaded with default/no embedding function.")
            except Exception as e:
                print(f" - Failed to load with no embedding function: {e}")
                
            # Try with SentenceTransformer
            try:
                ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                c = client.get_collection(col.name, embedding_function=ef)
                print(" - Successfully loaded with SentenceTransformer embedding function.")
            except Exception as e:
                print(f" - Failed to load with SentenceTransformer: {e}")

    except Exception as e:
        print(f"Error connecting to Chroma: {e}")

if __name__ == "__main__":
    check_collections()
