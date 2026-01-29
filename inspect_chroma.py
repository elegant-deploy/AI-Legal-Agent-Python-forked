"""
Inspect how documents are stored and chunked in ChromaDB.
Run: python inspect_chroma.py
No RAG pipeline code — inspection only.
"""
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from config.settings import settings


def create_chroma_client():
    """Same client as ingest — Chroma Cloud."""
    return chromadb.CloudClient(
        api_key=settings.CHROMA_API_KEY,
        tenant=settings.CHROMA_TENANT,
        database=settings.CHROMA_DATABASE,
    )


def inspect():
    client = create_chroma_client()

    print("=" * 60)
    print("CHROMADB INSPECTION — How documents are chunked & stored")
    print("=" * 60)

    try:
        collections = client.list_collections()
    except Exception as e:
        print(f"❌ Failed to list collections: {e}")
        return

    if not collections:
        print("No collections found. Ingest PDFs first.")
        return

    print(f"\n📚 Total collections: {len(collections)}\n")

    for coll in collections:
        name = coll.name
        try:
            col = client.get_collection(name=name)
            count = col.count()
        except Exception as e:
            print(f"❌ Collection '{name}': {e}\n")
            continue

        print("-" * 60)
        print(f"Collection: {name}")
        print(f"  Document (chunk) count: {count}")

        if count == 0:
            print()
            continue

        # Sample: get first 3 chunks (Chroma returns ids by default; include only documents, metadatas)
        try:
            sample = col.get(
                limit=3,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            print(f"  ⚠️ Could not get sample: {e}\n")
            continue

        ids = sample.get("ids", [])
        docs = sample.get("documents", [])
        metadatas = sample.get("metadatas", [])

        print(f"  Sample chunk IDs (first 3): {ids[:3]}")

        if metadatas and len(metadatas) > 0:
            first_meta = metadatas[0]
            print(f"  Metadata keys per chunk: {list(first_meta.keys())}")
            # Show values for first chunk
            print(f"  First chunk metadata: source={first_meta.get('source')}, page_number={first_meta.get('page_number')}, domain={first_meta.get('domain')}, chunk_type={first_meta.get('chunk_type')}")

        if docs and len(docs) > 0:
            first_doc = docs[0]
            preview = first_doc[:250].replace("\n", " ") if first_doc else ""
            print(f"  First chunk text preview ({len(first_doc)} chars): \"{preview}...\"")

        print()

    print("=" * 60)
    print("Summary of how ingest stores chunks (from ingest_controller):")
    print("  - Chunking (current): simple_chunking(chunk_size=600, chunk_overlap=100) — one pass")
    print("  - Older collections may still use smart_chunking (primary, sliding_window, semantic_split)")
    print("  - Each chunk: content (text) + metadata (source, page_number, domain, chunk_id, chunk_type, ...)")
    print("  - Embeddings: thenlper/gte-base (Deepinfra) when DEEPINFRA_API_KEY set; else BGE-M3 from HF")
    print("  - Stored in Chroma: ids, documents (text), metadatas, embeddings")
    print("  - ID format: {collection_name}_{pdf_hash}_{index:06d}")
    print("=" * 60)


if __name__ == "__main__":
    inspect()
