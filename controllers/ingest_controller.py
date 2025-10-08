import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from config.settings import settings

# Domain detection from filenames
DOMAIN_KEYWORDS = {
    'traffic': ['traffic', 'motor', 'vehicle', 'driving', 'license', 'transport', 'road'],
    'family': ['family', 'marriage', 'divorce', 'inheritance', 'guardian', 'child', 'maintenance', 'custody'],
    'corporate': ['corporate', 'company', 'business', 'commercial', 'contract', 'partnership', 'incorporation'],
    'ppc': ['ppc', 'penal', 'criminal', 'crime', 'offense', 'punishment', 'ipc', 'pakistan penal']
}

def detect_domain_from_filename(filename):
    """Detect legal domain from filename"""
    filename_lower = filename.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in filename_lower for keyword in keywords):
            return domain
    return 'general'  # Default domain

def create_chroma_client():
    """Create Chroma Cloud client"""
    client = chromadb.CloudClient(
        api_key=settings.CHROMA_API_KEY,
        tenant=settings.CHROMA_TENANT,
        database=settings.CHROMA_DATABASE
    )
    return client

def clean_text(text):
    """Clean and normalize text extracted from PDF"""
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Fix common PDF extraction issues
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)  # Fix hyphenated words
    # Remove excessive line breaks but keep paragraph structure
    text = re.sub(r'\n+', '\n', text)
    # Fix section numbers and titles
    text = re.sub(r'Section\s+(\d+)[.:]\s*', r'Section \1: ', text)
    # Clean up numbered lists
    text = re.sub(r'\((\d+)\)\s*', r'(\1) ', text)
    return text.strip()

def extract_text_with_metadata(pdf_path: str):
    """Extract text from PDF with metadata"""
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'")

    reader = PdfReader(str(p))
    documents = []

    # Detect domain from filename
    domain = detect_domain_from_filename(p.name)

    print(f"📖 Reading PDF: {p.name}")
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            cleaned_text = clean_text(text)
            if cleaned_text:
                documents.append({
                    'page_content': cleaned_text,
                    'metadata': {
                        'source': p.name,
                        'page_number': i + 1,
                        'document_type': 'legal',
                        'jurisdiction': 'Pakistan',
                        'source_type': 'pdf',
                        'domain': domain
                    }
                })
        # Show progress for large PDFs
        if (i + 1) % 10 == 0:
            print(f"   Processed page {i + 1}")

    if not documents:
        raise ValueError("No text extracted from PDF.")

    return documents, domain

def smart_chunking(documents, chunk_size=800, chunk_overlap=100):
    """Improved chunking that preserves legal document structure"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )

    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc['page_content'])
        for j, chunk in enumerate(chunks):
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata.update({
                'chunk_id': j,
                'total_chunks_doc': len(chunks),
                'chunk_size': len(chunk)
            })
            all_chunks.append({
                'content': chunk,
                'metadata': chunk_metadata
            })

    return all_chunks

def get_collection_name(domain, pdf_name):
    """Generate collection name based on domain"""
    base_name = pdf_name.lower().replace('.pdf', '').replace(' ', '_')
    return f"legal_docs_{domain}_{base_name}"

def list_existing_collections():
    """List all existing collections"""
    client = create_chroma_client()
    try:
        collections = client.list_collections()
        return collections
    except Exception as e:
        print(f"❌ Error listing collections: {e}")
        return []

def ingest_pdf(pdf_path: str, force: bool = False):
    """Main function to ingest PDF into ChromaDB"""
    client = create_chroma_client()

    # Extract documents and detect domain
    documents, domain = extract_text_with_metadata(pdf_path)
    collection_name = get_collection_name(domain, Path(pdf_path).name)

    print(f"\n🎯 Detected Domain: {domain.upper()}")
    print(f"📁 Collection Name: {collection_name}")

    # Check existing collection
    try:
        existing_collections = client.list_collections()
        existing_names = [col.name for col in existing_collections]

        if collection_name in existing_names and not force:
            return False, collection_name, "Collection already exists. Use force=True to recreate."
        elif collection_name in existing_names and force:
            print(f"🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"⚠️  Error checking collections: {e}")

    print(f"📖 Extracting and processing text from '{pdf_path}'...")
    print(f"📄 Extracted {len(documents)} pages with text")

    # Create chunks with improved chunking
    chunks = smart_chunking(documents, chunk_size=600, chunk_overlap=80)
    print(f"🔪 Created {len(chunks)} chunks")

    # Prepare data for Chroma
    documents_list = []
    metadatas_list = []
    ids_list = []

    for i, chunk_data in enumerate(chunks):
        documents_list.append(chunk_data['content'])
        metadatas_list.append(chunk_data['metadata'])
        ids_list.append(f"chunk_{i:04d}")

    # Compute embeddings
    print("🧮 Computing embeddings...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(documents_list, show_progress_bar=True, convert_to_numpy=True)

    # Create collection and upload
    print(f"🚀 Creating collection '{collection_name}'...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection(name=collection_name, embedding_function=ef)

    # Upload in batches
    batch_size = 100
    total_batches = (len(documents_list) + batch_size - 1) // batch_size

    for i in range(0, len(documents_list), batch_size):
        end_idx = min(i + batch_size, len(documents_list))
        batch_num = (i // batch_size) + 1

        print(f"📤 Uploading batch {batch_num}/{total_batches} ({end_idx-i} documents)...")

        collection.add(
            documents=documents_list[i:end_idx],
            embeddings=embeddings[i:end_idx].tolist(),
            metadatas=metadatas_list[i:end_idx],
            ids=ids_list[i:end_idx]
        )

    # Verify upload
    count = collection.count()
    print(f"✅ Upload complete! Collection '{collection_name}' has {count} documents.")

    return True, collection_name, f"Successfully ingested {count} chunks from {len(documents)} pages."