"""
Simple lexical index for hybrid retrieval.

This module provides a lightweight, dependency-free lexical search index
that complements vector search for exact-match queries (identifiers,
file names, table names, etc.).

Phase 2 Enhancement: Hybrid Search with Cheap Lexical Fallback
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from threading import RLock


# =============================================================================
# Identifier Detection
# =============================================================================

# Patterns that suggest a query is looking for an exact identifier
IDENTIFIER_PATTERNS = [
    r"\w+_\w+",           # underscore_separated (e.g., tbl_orders_001)
    r"\w+-\w+-\w+",       # hyphen-separated 3+ parts (e.g., my-config-file)
    r"[a-zA-Z]+\d+",      # letters+digits (e.g., user123, v2)
    r"\d+[a-zA-Z]+",      # digits+letters (e.g., 2nd, 3rd)
    r"\.\w{2,4}$",        # file extension (e.g., .py, .json, .md)
    r"\S{20,}",           # very long contiguous token
    r"^[A-Z][a-z]+[A-Z]", # CamelCase
    r"[a-z][A-Z]",        # camelCase mid-word
]


def looks_like_identifier(query: str) -> bool:
    """
    Detect if a query looks like an exact identifier.

    Identifiers are things like file names, table names, variable names,
    version numbers, etc. that benefit from exact lexical matching rather
    than semantic similarity.

    Args:
        query: The search query string

    Returns:
        True if query appears to be looking for an exact identifier

    Examples:
        >>> looks_like_identifier("tbl_orders_001")
        True
        >>> looks_like_identifier("my-config-file.json")
        True
        >>> looks_like_identifier("user123")
        True
        >>> looks_like_identifier("what is the meaning of life")
        False
    """
    # Check for any identifier pattern
    for pattern in IDENTIFIER_PATTERNS:
        if re.search(pattern, query):
            return True
    return False


# =============================================================================
# Lexical Index
# =============================================================================

class LexicalIndex:
    """
    Simple in-memory lexical search index.

    Uses inverted index (token -> doc_ids) for fast lookups.
    No external dependencies - just standard library.

    Thread-safe via RLock for concurrent access.

    Example:
        >>> idx = LexicalIndex()
        >>> idx.add("doc1", "The quick brown fox")
        >>> idx.add("doc2", "A quick brown dog")
        >>> idx.search("quick fox", k=2)
        [('doc1', 1.0), ('doc2', 0.5)]
    """

    def __init__(self):
        """Initialize empty lexical index."""
        self.documents: Dict[str, str] = {}      # id -> raw text
        self.tokens: Dict[str, Set[str]] = {}    # token -> set of doc ids
        self._lock = RLock()

    def add(self, doc_id: str, text: str) -> None:
        """
        Index a document.

        Args:
            doc_id: Unique identifier for the document
            text: Document text to index
        """
        with self._lock:
            # Remove old entry if exists
            if doc_id in self.documents:
                self._remove_doc_from_index(doc_id)

            self.documents[doc_id] = text
            for token in self._tokenize(text):
                if token not in self.tokens:
                    self.tokens[token] = set()
                self.tokens[token].add(doc_id)

    def remove(self, doc_id: str) -> bool:
        """
        Remove a document from the index.

        Args:
            doc_id: Document identifier to remove

        Returns:
            True if document was found and removed, False otherwise
        """
        with self._lock:
            if doc_id not in self.documents:
                return False
            self._remove_doc_from_index(doc_id)
            del self.documents[doc_id]
            return True

    def _remove_doc_from_index(self, doc_id: str) -> None:
        """Remove a document's tokens from the inverted index."""
        text = self.documents.get(doc_id, "")
        for token in self._tokenize(text):
            if token in self.tokens:
                self.tokens[token].discard(doc_id)
                # Clean up empty sets
                if not self.tokens[token]:
                    del self.tokens[token]

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search by token overlap.

        Scores documents by the fraction of query tokens they contain.

        Args:
            query: Search query
            k: Maximum number of results to return

        Returns:
            List of (doc_id, score) tuples, sorted by score descending.
            Score is in range [0.0, 1.0] representing fraction of query tokens matched.
        """
        with self._lock:
            query_tokens = set(self._tokenize(query))
            if not query_tokens:
                return []

            scores: Dict[str, int] = {}
            for token in query_tokens:
                for doc_id in self.tokens.get(token, []):
                    scores[doc_id] = scores.get(doc_id, 0) + 1

            # Normalize by query length
            results = [
                (doc_id, score / len(query_tokens))
                for doc_id, score in scores.items()
            ]
            return sorted(results, key=lambda x: -x[1])[:k]

    def search_exact(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for exact substring matches.

        More aggressive than token overlap - looks for the query
        as a substring in documents. Useful for identifiers.

        Args:
            query: Search query (will be searched as substring)
            k: Maximum number of results to return

        Returns:
            List of (doc_id, score) tuples. Score is 1.0 for exact match,
            0.5 for case-insensitive match.
        """
        with self._lock:
            query_lower = query.lower()
            results: List[Tuple[str, float]] = []

            for doc_id, text in self.documents.items():
                # Exact match
                if query in text:
                    results.append((doc_id, 1.0))
                # Case-insensitive match
                elif query_lower in text.lower():
                    results.append((doc_id, 0.8))

            return sorted(results, key=lambda x: -x[1])[:k]

    def get(self, doc_id: str) -> Optional[str]:
        """
        Retrieve document text by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document text or None if not found
        """
        with self._lock:
            return self.documents.get(doc_id)

    def __len__(self) -> int:
        """Return number of indexed documents."""
        with self._lock:
            return len(self.documents)

    def __contains__(self, doc_id: str) -> bool:
        """Check if document is in index."""
        with self._lock:
            return doc_id in self.documents

    def clear(self) -> None:
        """Clear all documents from the index."""
        with self._lock:
            self.documents.clear()
            self.tokens.clear()

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save index to a JSON file.

        Args:
            path: File path to save to (should end in .json)
        """
        with self._lock:
            data = {
                "documents": self.documents,
                # Convert sets to lists for JSON serialization
                "tokens": {k: list(v) for k, v in self.tokens.items()},
            }
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> bool:
        """
        Load index from a JSON file.

        Args:
            path: File path to load from

        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        path_obj = Path(path)
        if not path_obj.exists():
            return False

        with self._lock:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.documents = data.get("documents", {})
            # Convert lists back to sets
            self.tokens = {k: set(v) for k, v in data.get("tokens", {}).items()}
            return True

    # =========================================================================
    # Tokenization
    # =========================================================================

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into searchable tokens.

        Uses simple word boundary splitting plus some normalization.
        Keeps underscores and hyphens in tokens for identifier matching.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase tokens
        """
        # Split on whitespace but keep underscores/hyphens in tokens
        # This helps with identifiers like "tbl_orders_001"
        tokens = re.findall(r'[\w\-]+', text.lower())

        # Also add split versions for compound identifiers
        # e.g., "tbl_orders_001" -> ["tbl_orders_001", "tbl", "orders", "001"]
        expanded = []
        for token in tokens:
            expanded.append(token)
            # Split on underscores
            if '_' in token:
                expanded.extend(token.split('_'))
            # Split on hyphens
            if '-' in token:
                expanded.extend(token.split('-'))
            # Split camelCase
            camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', token)
            if len(camel_parts) > 1:
                expanded.extend(p.lower() for p in camel_parts)

        return [t for t in expanded if t]  # Filter empty strings


# =============================================================================
# Result Merging
# =============================================================================

def merge_search_results(
    vector_results: List[Tuple[str, float]],
    lexical_results: List[Tuple[str, float]],
    alpha: float = 0.7,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Merge vector and lexical search results.

    Uses weighted combination of scores.

    Args:
        vector_results: List of (doc_id, score) from vector search
        lexical_results: List of (doc_id, score) from lexical search
        alpha: Weight for vector scores (1-alpha for lexical)
        k: Maximum results to return

    Returns:
        Merged list of (doc_id, combined_score) tuples
    """
    # Convert to dicts for easy lookup
    vector_scores = dict(vector_results)
    lexical_scores = dict(lexical_results)

    # Get all unique doc IDs
    all_ids = set(vector_scores.keys()) | set(lexical_scores.keys())

    # Combine scores
    combined = []
    for doc_id in all_ids:
        v_score = vector_scores.get(doc_id, 0.0)
        l_score = lexical_scores.get(doc_id, 0.0)
        combined_score = alpha * v_score + (1 - alpha) * l_score
        combined.append((doc_id, combined_score))

    # Sort by combined score and return top k
    return sorted(combined, key=lambda x: -x[1])[:k]
