import os
import re
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from config.settings import settings
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub import InferenceClient

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

    # Compute PDF hash for deterministic IDs
    with open(pdf_path, 'rb') as f:
        pdf_hash = hashlib.sha256(f.read()).hexdigest()

    reader = PdfReader(str(p))
    documents = []

    # Detect domain from filename
    domain = detect_domain_from_filename(p.name)

    print(f"📖 Reading PDF: {p.name} (hash: {pdf_hash[:8]}...)")
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
                        'domain': domain,
                        'pdf_hash': pdf_hash
                    }
                })
        # Show progress for large PDFs
        if (i + 1) % 10 == 0:
            print(f"   Processed page {i + 1}")

    if not documents:
        raise ValueError("No text extracted from PDF.")

    return documents, domain, pdf_hash


def simple_chunking(documents, chunk_size=600, chunk_overlap=100):
    """One-pass chunking only — fewer chunks, no sliding window or semantic split."""
    legal_separators = [
        "\n\nSection", "\n\nArticle", "\n\nChapter", "\n\nPart", "\n\nClause",
        "\n\n(", "\n\n1.", "\n\n2.", "\n\n3.", "\n\n4.", "\n\n5.", "\n\n6.", "\n\n7.", "\n\n8.", "\n\n9.", "\n\n10.",
        "\n\n(a)", "\n\n(b)", "\n\n(c)", "\n\n(d)", "\n\n(e)", "\n\n(f)", "\n\n(g)", "\n\n(h)", "\n\n(i)", "\n\n(j)",
        "\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=legal_separators,
        keep_separator=True,
    )
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc["page_content"])
        for j, chunk in enumerate(doc_chunks):
            if len(chunk.strip()) >= 50:
                meta = doc["metadata"].copy()
                meta.update({
                    "chunk_id": j,
                    "total_chunks_doc": len(doc_chunks),
                    "chunk_size": len(chunk),
                    "chunk_start_pos": j * (chunk_size - chunk_overlap),
                    "has_section_header": any(h in chunk[:150] for h in ["Section", "Article", "Chapter", "Part", "Clause"]),
                    "chunk_type": "primary",
                })
                chunks.append({"content": chunk, "metadata": meta})
    return chunks


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

def get_deepinfra_embeddings(texts, api_key, model):
    """Get embeddings from Deepinfra (OpenAI-compatible API), e.g. thenlper/gte-base."""
    url = "https://api.deepinfra.com/v1/openai/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "input": texts if isinstance(texts, list) else [texts], "encoding_format": "float"}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    out = [item["embedding"] for item in data["data"]]
    return [list(map(float, emb)) for emb in out]


def get_hf_embeddings(texts, api_key, model):
    """Get embeddings from Hugging Face Inference API"""
    try:
        client = InferenceClient(model=model, token=api_key)
        embeddings = client.feature_extraction(texts)
        # Ensure embeddings are list of lists of floats
        return [list(map(float, emb)) for emb in embeddings]
    except Exception as e:
        print(f"Error calling HF Inference API: {e}")
        raise


def compute_embeddings(chunks):
    """Compute embeddings for chunks. Uses Deepinfra if DEEPINFRA_API_KEY set, else HF."""
    texts = [chunk["content"] for chunk in chunks]
    print(f"🧮 Computing embeddings for {len(texts)} chunks...")
    batch_size = settings.EMBED_BATCH_SIZE
    total_batches = (len(texts) + batch_size - 1) // batch_size
    embeddings = []

    use_deepinfra = getattr(settings, "DEEPINFRA_API_KEY", None) and settings.DEEPINFRA_API_KEY.strip()
    if use_deepinfra:
        api_key = settings.DEEPINFRA_API_KEY
        model = getattr(settings, "DEEPINFRA_EMBED_MODEL", None) or "thenlper/gte-base"
        print(f"   Using Deepinfra: {model}")
        get_embs = get_deepinfra_embeddings
    else:
        api_key = settings.HF_API_KEY
        model = settings.HF_EMBED_MODEL
        print(f"   Using Hugging Face: {model}")
        get_embs = get_hf_embeddings

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"   📤 Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
        try:
            batch_embs = get_embs(batch_texts, api_key, model)
            batch_embs = [list(emb) for emb in batch_embs]
            embeddings.extend(batch_embs)
        except Exception as e:
            print(f"❌ Error computing batch {batch_num}: {e}")
            raise

    return embeddings

def ingest_pdf(pdf_path: str, force: bool = False, dry_run: bool = False):
    """Main function to ingest PDF into ChromaDB with BGE-M3 embeddings"""
    client = create_chroma_client()

    # Extract documents and detect domain
    documents, domain, pdf_hash = extract_text_with_metadata(pdf_path)
    collection_name = get_collection_name(domain, Path(pdf_path).name)

    print(f"\n🎯 Detected Domain: {domain.upper()}")
    print(f"📁 Collection Name: {collection_name}")

    # Check existing collection
    try:
        existing_collections = client.list_collections()
        existing_names = [col.name for col in existing_collections]

        if collection_name in existing_names and not force and not dry_run:
            return False, collection_name, "Collection already exists. Use force=True to recreate."
        elif collection_name in existing_names and force and not dry_run:
            print(f"🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(name=collection_name)
    except Exception as e:
        print(f"⚠️  Error checking collections: {e}")

    print(f"📖 Extracting and processing text from '{pdf_path}'...")
    print(f"📄 Extracted {len(documents)} pages with text")

    # Simple one-pass chunking — fewer chunks (no sliding window / semantic split)
    chunks = simple_chunking(documents, chunk_size=600, chunk_overlap=100)
    avg_len = sum(len(c["content"]) for c in chunks) // len(chunks) if chunks else 0
    print(f"🔪 Created {len(chunks)} chunks (avg size: {avg_len} chars)")

    # Prepare data for Chroma
    documents_list = []
    metadatas_list = []
    ids_list = []

    for i, chunk_data in enumerate(chunks):
        documents_list.append(chunk_data['content'])
        metadatas_list.append(chunk_data['metadata'])
        # Deterministic unique ID: collection_name + pdf_hash + global_index
        ids_list.append(f"{collection_name}_{pdf_hash}_{i:06d}")

    # Compute embeddings
    embeddings_list = compute_embeddings(chunks)

    if dry_run:
        print(f"🔍 Dry run: Would upload {len(embeddings_list)} embeddings to collection '{collection_name}'")
        return True, collection_name, f"Dry run complete for {len(embeddings_list)} chunks"

    # Create collection and upload
    print(f"🚀 Creating collection '{collection_name}'...")
    collection = client.create_collection(name=collection_name)

    # Upload in batches
    batch_size = 100
    total_batches = (len(documents_list) + batch_size - 1) // batch_size

    for i in range(0, len(documents_list), batch_size):
        end_idx = min(i + batch_size, len(documents_list))
        batch_num = (i // batch_size) + 1

        print(f"📤 Uploading batch {batch_num}/{total_batches} ({end_idx-i} documents)...")

        try:
            collection.add(
                documents=documents_list[i:end_idx],
                embeddings=embeddings_list[i:end_idx],
                metadatas=metadatas_list[i:end_idx],
                ids=ids_list[i:end_idx]
            )
        except Exception as e:
            print(f"❌ Error uploading batch {batch_num}: {e}")
            raise

    # Verify upload
    count = collection.count()
    print(f"✅ Upload complete! Collection '{collection_name}' has {count} documents.")

    return True, collection_name, f"Successfully ingested {count} chunks from {len(documents)} pages."