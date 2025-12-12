"""
File access control for Sema (Semantic Memory Agent).

Provides a whitelist-based file access system that:
- Reads allowed paths from sema_files.txt
- Expands directory entries to their contained files
- Validates file access requests against the whitelist
- Supports text files, PDFs, and Word documents
- Provides an agentic file_read tool for LLM tool calling
"""

import os
from pathlib import Path
from typing import Set, List, Dict, Optional, Any, Tuple
from dataclasses import dataclass

# Base directory is the repo root (parent of sem_mem/)
BASE_DIR = Path(__file__).resolve().parent.parent
WHITELIST_FILE = BASE_DIR / "sema_files.txt"

# File extensions we support
TEXT_EXTENSIONS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
    ".sh", ".bash", ".zsh",
    ".sql", ".graphql",
    ".xml", ".csv",
    ".ini", ".cfg", ".conf",
    ".gitignore", ".env", ".example",
}

BINARY_EXTENSIONS = {
    ".pdf",
    ".doc", ".docx",  # Word documents
}

# Extensions to always exclude (even in whitelisted directories)
EXCLUDED_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll",
    ".bin", ".pkl", ".pickle", ".npy", ".npz",
    ".h5", ".hdf5", ".ckpt", ".pt", ".pth",  # Model checkpoints
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".exe", ".app",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg",  # Images (unless explicitly needed)
    ".mp3", ".mp4", ".wav", ".avi", ".mov",  # Media
}

# Directories to always exclude
EXCLUDED_DIRS = {
    "__pycache__", ".git", ".svn", ".hg",
    "node_modules", ".venv", "venv", "env",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".eggs", "*.egg-info", "dist", "build",
    ".idea", ".vscode",
}


def _is_hidden(path: Path) -> bool:
    """Check if a path or any of its components is hidden (starts with .)."""
    return any(part.startswith(".") and part not in {".env", ".example"} for part in path.parts)


def _is_excluded_dir(name: str) -> bool:
    """Check if a directory name should be excluded."""
    return name in EXCLUDED_DIRS or name.startswith(".")


def _is_excluded_file(path: Path) -> bool:
    """Check if a file should be excluded based on extension."""
    return path.suffix.lower() in EXCLUDED_EXTENSIONS


def _normalize_path(path_str: str) -> str:
    """Normalize a path string to posix-style relative path."""
    # Remove leading/trailing whitespace and slashes
    path_str = path_str.strip().strip("/").strip("\\")
    # Convert to posix style
    return path_str.replace("\\", "/")


def load_whitelist_raw() -> List[str]:
    """
    Load raw entries from the whitelist file.

    Returns list of non-empty, non-comment lines as-is.
    Useful for UI display.
    """
    if not WHITELIST_FILE.exists():
        return []

    entries = []
    with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)

    return entries


def load_whitelist() -> Set[str]:
    """
    Load and expand the whitelist to a set of allowed file paths.

    - Reads sema_files.txt from BASE_DIR
    - Expands directory entries to all contained files (recursively)
    - Excludes hidden files/dirs and binary artifacts
    - Returns normalized posix-style relative paths from BASE_DIR

    Returns:
        Set of allowed relative file paths (e.g., {"sem_mem/core.py", "README.md"})
    """
    if not WHITELIST_FILE.exists():
        return set()

    allowed_files: Set[str] = set()

    with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            entry = _normalize_path(line)
            full_path = (BASE_DIR / entry).resolve()

            # Security: ensure path is under BASE_DIR
            try:
                full_path.relative_to(BASE_DIR)
            except ValueError:
                # Path escapes BASE_DIR, skip it
                continue

            if full_path.is_file():
                # Direct file entry
                if not _is_excluded_file(full_path):
                    rel_path = str(full_path.relative_to(BASE_DIR)).replace("\\", "/")
                    allowed_files.add(rel_path)

            elif full_path.is_dir():
                # Directory entry - expand recursively
                for root, dirs, files in os.walk(full_path):
                    root_path = Path(root)

                    # Filter out excluded directories (modifies dirs in-place)
                    dirs[:] = [d for d in dirs if not _is_excluded_dir(d)]

                    for filename in files:
                        file_path = root_path / filename

                        # Skip hidden and excluded files
                        if filename.startswith(".") and filename not in {".env.example"}:
                            continue
                        if _is_excluded_file(file_path):
                            continue

                        rel_path = str(file_path.relative_to(BASE_DIR)).replace("\\", "/")
                        allowed_files.add(rel_path)

    return allowed_files


def get_file_info(allowed_files: Optional[Set[str]] = None) -> List[Dict]:
    """
    Get information about all allowed files.

    Args:
        allowed_files: Pre-loaded whitelist set, or None to load fresh

    Returns:
        List of dicts with file information:
        [
            {"path": "sem_mem/core.py", "size": 1234, "content_type": "text"},
            {"path": "docs/guide.pdf", "size": 5678, "content_type": "pdf"},
            ...
        ]
    """
    if allowed_files is None:
        allowed_files = load_whitelist()

    files_info = []

    for rel_path in sorted(allowed_files):
        full_path = BASE_DIR / rel_path

        if not full_path.exists():
            continue

        # Determine content type
        suffix = full_path.suffix.lower()
        if suffix == ".pdf":
            content_type = "pdf"
        elif suffix in {".doc", ".docx"}:
            content_type = "word"
        elif suffix in TEXT_EXTENSIONS or suffix == "":
            content_type = "text"
        else:
            content_type = "binary"

        try:
            size = full_path.stat().st_size
        except OSError:
            size = 0

        files_info.append({
            "path": rel_path,
            "size": size,
            "content_type": content_type,
        })

    return files_info


def validate_file_access(path: str, allowed_files: Optional[Set[str]] = None) -> Path:
    """
    Validate that a requested path is allowed and exists.

    Args:
        path: Relative path requested (e.g., "sem_mem/core.py")
        allowed_files: Pre-loaded whitelist set, or None to load fresh

    Returns:
        Resolved absolute Path to the file

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file is not in whitelist
    """
    if allowed_files is None:
        allowed_files = load_whitelist()

    # Normalize the requested path
    normalized = _normalize_path(path)

    # Resolve to absolute path
    requested = (BASE_DIR / normalized).resolve()

    # Security: ensure path is under BASE_DIR
    try:
        requested.relative_to(BASE_DIR)
    except ValueError:
        raise PermissionError(f"Access denied: path escapes base directory")

    # Check existence
    if not requested.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    # Check whitelist
    rel_path = str(requested.relative_to(BASE_DIR)).replace("\\", "/")
    if rel_path not in allowed_files:
        raise PermissionError(f"Access denied: {path} is not in whitelist")

    return requested


def read_file_content(path: Path) -> tuple:
    """
    Read file content based on its type.

    Args:
        path: Absolute path to the file

    Returns:
        Tuple of (content, media_type):
        - For text files: (str, "text/plain")
        - For PDFs: (bytes, "application/pdf")
        - For Word docs: (bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return path.read_bytes(), "application/pdf"

    elif suffix == ".docx":
        return path.read_bytes(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    elif suffix == ".doc":
        return path.read_bytes(), "application/msword"

    else:
        # Text file - read with UTF-8, ignore errors
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except UnicodeDecodeError:
            # Fallback: read as bytes and decode
            content = path.read_bytes().decode("utf-8", errors="ignore")
        return content, "text/plain; charset=utf-8"


# =============================================================================
# Whitelist Management (for UI)
# =============================================================================

def add_to_whitelist(entry: str) -> bool:
    """
    Add an entry to the whitelist file.

    Args:
        entry: Path to add (file or directory)

    Returns:
        True if added, False if already present
    """
    entry = entry.strip()
    if not entry:
        return False

    # Load current entries
    current_entries = set(load_whitelist_raw())

    if entry in current_entries:
        return False

    # Append to file
    with open(WHITELIST_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{entry}")

    return True


def remove_from_whitelist(entry: str) -> bool:
    """
    Remove an entry from the whitelist file.

    Args:
        entry: Path to remove

    Returns:
        True if removed, False if not found
    """
    entry = entry.strip()
    if not entry:
        return False

    if not WHITELIST_FILE.exists():
        return False

    # Read all lines
    with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out the entry
    new_lines = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped == entry:
            found = True
        else:
            new_lines.append(line)

    if not found:
        return False

    # Write back
    with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return True


def get_suggested_paths() -> List[str]:
    """
    Get a list of suggested paths that could be added to whitelist.

    Returns common directories and important files that aren't already whitelisted.
    """
    current = set(load_whitelist_raw())

    suggestions = []

    # Common directories
    for dirname in ["sem_mem/", "tests/", "docs/", "scripts/"]:
        dir_path = BASE_DIR / dirname.rstrip("/")
        if dir_path.is_dir() and dirname not in current:
            suggestions.append(dirname)

    # Important root files
    for filename in ["app.py", "server.py", "README.md", "CLAUDE.md", "pyproject.toml"]:
        if (BASE_DIR / filename).is_file() and filename not in current:
            suggestions.append(filename)

    return suggestions


# =============================================================================
# Agentic File Read Tool (LLM Tool Calling)
# =============================================================================

@dataclass
class FileReadResult:
    """Result from reading a file."""
    path: str
    content: str
    content_type: str
    size: int
    success: bool
    error: Optional[str] = None


def fetch_file(path: str, allowed_files: Optional[Set[str]] = None, max_chars: int = 50000) -> FileReadResult:
    """
    Read a whitelisted file and return its content.

    This is the agentic interface for file reading - the LLM can request
    specific files and receive their content.

    Args:
        path: Relative path to the file (e.g., "sem_mem/core.py")
        allowed_files: Pre-loaded whitelist, or None to load fresh
        max_chars: Maximum characters to return (truncates if exceeded)

    Returns:
        FileReadResult with content or error
    """
    try:
        # Validate access
        full_path = validate_file_access(path, allowed_files)

        # Read content
        content, media_type = read_file_content(full_path)

        # Handle binary content (PDF, Word docs)
        if isinstance(content, bytes):
            # For PDFs, try to extract text
            if media_type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(content))
                    text_parts = []
                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    content = "\n\n".join(text_parts)
                    content_type = "pdf (extracted text)"
                except Exception as e:
                    return FileReadResult(
                        path=path,
                        content="",
                        content_type="pdf",
                        size=len(content),
                        success=False,
                        error=f"Could not extract PDF text: {e}",
                    )
            else:
                # Word docs or other binary - not yet supported for text extraction
                return FileReadResult(
                    path=path,
                    content="",
                    content_type=media_type,
                    size=len(content),
                    success=False,
                    error=f"Binary file type not supported for text extraction: {media_type}",
                )
        else:
            content_type = "text"

        # Truncate if too long
        original_size = len(content)
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[Content truncated - {original_size} total chars]"

        return FileReadResult(
            path=path,
            content=content,
            content_type=content_type,
            size=original_size,
            success=True,
        )

    except FileNotFoundError as e:
        return FileReadResult(
            path=path,
            content="",
            content_type="",
            size=0,
            success=False,
            error=f"File not found: {path}",
        )
    except PermissionError as e:
        return FileReadResult(
            path=path,
            content="",
            content_type="",
            size=0,
            success=False,
            error=str(e),
        )
    except Exception as e:
        return FileReadResult(
            path=path,
            content="",
            content_type="",
            size=0,
            success=False,
            error=f"Error reading file: {e}",
        )


def format_file_read_result(result: FileReadResult) -> str:
    """
    Format a file read result as context for the LLM.

    Args:
        result: FileReadResult from fetch_file()

    Returns:
        Formatted string for injection into prompt context
    """
    if not result.success:
        return f"Failed to read {result.path}: {result.error}"

    return f"""Content of {result.path} ({result.content_type}, {result.size} chars):

{result.content}"""


def get_file_read_tool_definition(api_format: str = "responses") -> Dict[str, Any]:
    """
    Get the OpenAI function tool definition for file_read.

    This allows the LLM to proactively request file content from whitelisted files.

    Args:
        api_format: Either "responses" (OpenAI Responses API) or "completions" (Chat Completions).

    Returns:
        Tool definition dict for the specified API format.
    """
    description = (
        "Read the content of a whitelisted local file. Use this when you need to see "
        "the actual content of a file in the codebase. You can only read files that "
        "have been added to the whitelist. The path should be relative to the repository root "
        "(e.g., 'sem_mem/core.py' or 'README.md')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The relative path to the file (e.g., 'sem_mem/core.py', 'app.py')",
            },
        },
        "required": ["path"],
    }

    if api_format == "responses":
        # OpenAI Responses API format (flat structure)
        return {
            "type": "function",
            "name": "file_read",
            "description": description,
            "parameters": parameters,
            "strict": False,
        }
    else:
        # Chat Completions API format (nested under "function")
        return {
            "type": "function",
            "function": {
                "name": "file_read",
                "description": description,
                "parameters": parameters,
            },
        }
