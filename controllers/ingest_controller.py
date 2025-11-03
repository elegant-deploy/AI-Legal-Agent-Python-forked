import os
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from config.settings import settings
import numpy as np

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

def smart_chunking(documents, chunk_size=500, chunk_overlap=150):
    """Advanced sliding window chunking that preserves legal document structure and improves retrieval"""
    # Enhanced legal-specific separators for better chunking
    legal_separators = [
        "\n\nSection", "\n\nArticle", "\n\nChapter", "\n\nPart", "\n\nClause",
        "\n\n(", "\n\n1.", "\n\n2.", "\n\n3.", "\n\n4.", "\n\n5.", "\n\n6.", "\n\n7.", "\n\n8.", "\n\n9.", "\n\n10.",
        "\n\n(a)", "\n\n(b)", "\n\n(c)", "\n\n(d)", "\n\n(e)", "\n\n(f)", "\n\n(g)", "\n\n(h)", "\n\n(i)", "\n\n(j)",
        "\n\n(i)", "\n\n(ii)", "\n\n(iii)", "\n\n(iv)", "\n\n(v)", "\n\n(vi)", "\n\n(vii)",
        "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
    ]

    # First pass: Create initial chunks with legal structure awareness
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=legal_separators,
        keep_separator=True
    )

    initial_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc['page_content'])
        for j, chunk in enumerate(chunks):
            if len(chunk.strip()) >= 50:  # Minimum meaningful chunk size
                chunk_metadata = doc['metadata'].copy()
                chunk_metadata.update({
                    'chunk_id': j,
                    'total_chunks_doc': len(chunks),
                    'chunk_size': len(chunk),
                    'chunk_start_pos': j * (chunk_size - chunk_overlap),
                    'has_section_header': any(header in chunk[:150] for header in ['Section', 'Article', 'Chapter', 'Part', 'Clause']),
                    'chunk_type': 'primary'
                })
                initial_chunks.append({
                    'content': chunk,
                    'metadata': chunk_metadata
                })

    # Second pass: Sliding window enhancement for better context overlap
    enhanced_chunks = []
    window_size = 3  # Number of chunks to consider in sliding window

    for i, chunk in enumerate(initial_chunks):
        enhanced_chunks.append(chunk)

        # Create sliding window overlaps for better retrieval
        if i > 0 and i < len(initial_chunks) - 1:
            # Create overlapping chunks with neighboring content
            prev_chunk = initial_chunks[i-1]['content'][-200:] if i > 0 else ""
            next_chunk = initial_chunks[i+1]['content'][:200] if i < len(initial_chunks)-1 else ""

            # Enhanced chunk with context from neighbors
            enhanced_content = prev_chunk + " " + chunk['content'] + " " + next_chunk
            enhanced_content = enhanced_content.strip()

            if len(enhanced_content) > chunk_size * 0.8:  # Only if meaningful enhancement
                enhanced_metadata = chunk['metadata'].copy()
                enhanced_metadata.update({
                    'chunk_type': 'sliding_window',
                    'window_size': window_size,
                    'has_context_overlap': True,
                    'chunk_size': len(enhanced_content)
                })

                enhanced_chunks.append({
                    'content': enhanced_content,
                    'metadata': enhanced_metadata
                })

    # Third pass: Semantic chunking for very long sections
    final_chunks = []
    for chunk in enhanced_chunks:
        content = chunk['content']
        if len(content) > chunk_size * 2:  # If chunk is too long, split semantically
            semantic_chunks = _semantic_split_long_chunk(content, chunk_size, chunk_overlap)
            for semantic_chunk in semantic_chunks:
                semantic_metadata = chunk['metadata'].copy()
                semantic_metadata.update({
                    'chunk_type': 'semantic_split',
                    'original_chunk_size': len(content),
                    'chunk_size': len(semantic_chunk)
                })
                final_chunks.append({
                    'content': semantic_chunk,
                    'metadata': semantic_metadata
                })
        else:
            final_chunks.append(chunk)

    return final_chunks

def _semantic_split_long_chunk(text, chunk_size, overlap):
    """Split long chunks based on semantic boundaries"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk + sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from previous chunk
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_text + " " + sentence + " "
            else:
                chunks.append(sentence)
                current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

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

    # Create chunks with advanced sliding window chunking for legal documents
    chunks = smart_chunking(documents, chunk_size=450, chunk_overlap=150)
    print(f"🔪 Created {len(chunks)} advanced chunks (avg size: {sum(len(c['content']) for c in chunks)//len(chunks) if chunks else 0} chars)")
    print(f"   📊 Chunk types: Primary={sum(1 for c in chunks if c['metadata'].get('chunk_type')=='primary')}, "
          f"Sliding={sum(1 for c in chunks if c['metadata'].get('chunk_type')=='sliding_window')}, "
          f"Semantic={sum(1 for c in chunks if c['metadata'].get('chunk_type')=='semantic_split')}")

    # Prepare data for Chroma
    documents_list = []
    metadatas_list = []
    ids_list = []

    for i, chunk_data in enumerate(chunks):
        documents_list.append(chunk_data['content'])
        metadatas_list.append(chunk_data['metadata'])
        ids_list.append(f"chunk_{i:04d}")

    try:
        # Try to use Qwen model if available, fallback to MiniLM
        embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("   Using all-MiniLM-L6-v2 for embeddings")
    except Exception as e:
        print(f"   Warning: Could not load preferred embedder: {e}")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embedder.encode(documents_list, show_progress_bar=True, convert_to_numpy=True)

    # Create collection and upload with optimized embedding function
    print(f"🚀 Creating collection '{collection_name}'...")
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("   Using all-MiniLM-L6-v2 embedding function")
    except Exception as e:
        print(f"   Warning: Could not load preferred embedding function: {e}")
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