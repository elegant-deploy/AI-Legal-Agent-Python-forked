#!/usr/bin/env python3
"""
Re-ingest all legal documents with BGE-M3 embeddings.

This script deletes all existing collections and re-ingests all PDFs in the Laws/ directory
using the new BGE-M3 embeddings and deterministic IDs.
"""

import os
import sys
from pathlib import Path
from controllers.ingest_controller import ingest_pdf

def main():
    """Re-ingest all PDFs in the Laws directory"""
    laws_dir = Path("Laws")

    if not laws_dir.exists():
        print("❌ Laws directory not found!")
        sys.exit(1)

    pdf_files = list(laws_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in Laws directory!")
        sys.exit(1)

    print(f"📚 Found {len(pdf_files)} PDF files to re-ingest:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    print("\n🚀 Starting re-ingestion with BGE-M3 embeddings...")

    success_count = 0
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"📖 Processing: {pdf_path.name}")
        print(f"{'='*60}")

        try:
            success, collection_name, message = ingest_pdf(str(pdf_path), force=True)
            if success:
                print(f"✅ Successfully ingested: {collection_name}")
                success_count += 1
            else:
                print(f"❌ Failed to ingest {pdf_path.name}: {message}")
        except Exception as e:
            print(f"❌ Error ingesting {pdf_path.name}: {e}")

    print(f"\n{'='*60}")
    print(f"📊 Re-ingestion complete: {success_count}/{len(pdf_files)} files successful")
    print(f"{'='*60}")

    if success_count == len(pdf_files):
        print("🎉 All documents re-ingested successfully with BGE-M3 embeddings!")
    else:
        print("⚠️ Some documents failed to re-ingest. Check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()