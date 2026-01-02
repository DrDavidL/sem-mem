#!/usr/bin/env python3
"""
Migrate HNSW index to a new embedding model.

This script re-embeds all memories using a new embedding provider/model,
creating a fresh HNSW index. The old index is backed up before migration.

Usage:
    # Migrate to local Qwen3 embeddings (recommended)
    python scripts/migrate_embeddings.py --provider sentence-transformers

    # Migrate with explicit model
    python scripts/migrate_embeddings.py --provider sentence-transformers --model Qwen/Qwen3-Embedding-0.6B

    # Dry run (show what would be migrated without changing anything)
    python scripts/migrate_embeddings.py --provider sentence-transformers --dry-run

    # Custom storage directory
    python scripts/migrate_embeddings.py --provider sentence-transformers --storage-dir ./my_memory
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sem_mem.providers import get_embedding_provider, get_available_embedding_providers
from sem_mem.config import DEFAULT_EMBEDDING_MODELS


def load_existing_metadata(storage_dir: str) -> Tuple[Dict, List[Dict]]:
    """
    Load existing HNSW metadata.

    Returns:
        Tuple of (full_metadata, list_of_entries)
    """
    metadata_path = os.path.join(storage_dir, "hnsw_metadata.json")

    if not os.path.exists(metadata_path):
        print(f"No existing index found at {metadata_path}")
        return {}, []

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    entries = metadata.get('entries', {})
    # Convert to list with IDs
    entry_list = [
        {"id": int(k), **v}
        for k, v in entries.items()
    ]

    return metadata, entry_list


def backup_index(storage_dir: str) -> str:
    """
    Create a backup of the existing index.

    Returns:
        Path to backup directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(storage_dir, f"backup_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    # Backup files
    files_to_backup = ["hnsw_index.bin", "hnsw_metadata.json", "lexical_index.json"]
    backed_up = []

    for filename in files_to_backup:
        src = os.path.join(storage_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(backup_dir, filename)
            shutil.copy2(src, dst)
            backed_up.append(filename)

    print(f"Backed up {len(backed_up)} files to {backup_dir}")
    return backup_dir


def migrate_embeddings(
    storage_dir: str,
    provider: str,
    model: Optional[str] = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> int:
    """
    Migrate all embeddings to a new provider/model.

    Args:
        storage_dir: Directory containing HNSW index
        provider: New embedding provider name
        model: New embedding model (uses provider default if not specified)
        batch_size: Number of texts to embed at once
        dry_run: If True, don't actually create new index

    Returns:
        Number of memories migrated
    """
    # Load existing metadata
    old_metadata, entries = load_existing_metadata(storage_dir)

    if not entries:
        print("No memories to migrate.")
        return 0

    old_provider = old_metadata.get('embedding_provider', 'unknown')
    old_model = old_metadata.get('embedding_model', 'unknown')
    old_dim = old_metadata.get('embedding_dim', 'unknown')

    print(f"\nExisting index:")
    print(f"  Provider: {old_provider}")
    print(f"  Model: {old_model}")
    print(f"  Dimension: {old_dim}")
    print(f"  Memories: {len(entries)}")

    # Get new embedding provider
    model = model or DEFAULT_EMBEDDING_MODELS.get(provider)
    print(f"\nNew embedding configuration:")
    print(f"  Provider: {provider}")
    print(f"  Model: {model}")

    if dry_run:
        print("\n[DRY RUN] Would migrate the following memories:")
        for i, entry in enumerate(entries[:5]):
            text = entry.get('text', '')[:80]
            print(f"  {i+1}. {text}...")
        if len(entries) > 5:
            print(f"  ... and {len(entries) - 5} more")
        return len(entries)

    # Create embedding provider
    print(f"\nInitializing {provider} embedding provider...")
    embedding_provider = get_embedding_provider(provider)
    new_dim = embedding_provider.default_dimension
    print(f"  Dimension: {new_dim}")

    # Backup existing index
    print("\nBacking up existing index...")
    backup_dir = backup_index(storage_dir)

    # Re-embed all texts one at a time (safer for memory)
    print(f"\nRe-embedding {len(entries)} memories...")
    texts = [entry.get('text', '') for entry in entries]
    new_vectors = []

    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing {i + 1}/{len(texts)}...")

        embedding = embedding_provider.embed_single(text)
        new_vectors.append(embedding)

    # Create new HNSW index
    print("\nCreating new HNSW index...")
    import hnswlib

    new_index = hnswlib.Index(space='cosine', dim=new_dim)
    new_index.init_index(
        max_elements=max(100000, len(entries) * 2),
        ef_construction=200,
        M=16,
    )
    new_index.set_ef(50)

    # Add vectors to index
    ids = np.array([entry['id'] for entry in entries])
    vectors = np.array(new_vectors)
    new_index.add_items(vectors, ids)

    # Update entries with new vectors
    new_entries = {}
    for entry, new_vec in zip(entries, new_vectors):
        entry_id = entry['id']
        # Remove old 'id' key from entry dict
        entry_data = {k: v for k, v in entry.items() if k != 'id'}
        entry_data['vector'] = new_vec.tolist()
        new_entries[str(entry_id)] = entry_data

    # Build text_to_id mapping
    text_to_id = {
        entry.get('text', ''): entry['id']
        for entry in entries
    }

    # Find next ID
    next_id = max(entry['id'] for entry in entries) + 1

    # Create new metadata
    new_metadata = {
        'entries': new_entries,
        'text_to_id': text_to_id,
        'next_id': next_id,
        'embedding_provider': provider,
        'embedding_model': model,
        'embedding_dim': new_dim,
        'migrated_from': {
            'provider': old_provider,
            'model': old_model,
            'dim': old_dim,
            'timestamp': datetime.now().isoformat(),
            'backup_dir': backup_dir,
        },
    }

    # Save new index and metadata
    print("Saving new index...")
    new_index.save_index(os.path.join(storage_dir, "hnsw_index.bin"))

    with open(os.path.join(storage_dir, "hnsw_metadata.json"), 'w') as f:
        json.dump(new_metadata, f, indent=2)

    print(f"\nMigration complete!")
    print(f"  Migrated: {len(entries)} memories")
    print(f"  New dimension: {new_dim}")
    print(f"  Backup at: {backup_dir}")

    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate HNSW embeddings to a new provider/model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate to local Qwen3 embeddings
    python scripts/migrate_embeddings.py --provider sentence-transformers

    # Dry run to see what would be migrated
    python scripts/migrate_embeddings.py --provider local --dry-run
        """,
    )

    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help=f"Embedding provider. Available: {', '.join(get_available_embedding_providers())}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model (uses provider default if not specified)",
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        default="./local_memory",
        help="Storage directory containing HNSW index (default: ./local_memory)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of texts to embed at once (default: 32)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )

    args = parser.parse_args()

    # Validate provider
    available = get_available_embedding_providers()
    if args.provider not in available:
        print(f"Error: Unknown provider '{args.provider}'")
        print(f"Available providers: {', '.join(available)}")
        sys.exit(1)

    # Run migration
    count = migrate_embeddings(
        storage_dir=args.storage_dir,
        provider=args.provider,
        model=args.model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if count > 0 and not args.dry_run:
        print("\nNext steps:")
        print("  1. Update your .env file:")
        print(f"     SEMMEM_EMBEDDING_PROVIDER={args.provider}")
        if args.model:
            print(f"     SEMMEM_EMBEDDING_MODEL={args.model}")
        print("  2. Restart your application")


if __name__ == "__main__":
    main()
