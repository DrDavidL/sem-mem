"""
FastAPI router for file access endpoints.

Provides whitelisted file access for Sema (Semantic Memory Agent).
"""

from typing import List
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from ..file_access import (
    load_whitelist,
    get_file_info,
    validate_file_access,
    read_file_content,
)


router = APIRouter(prefix="/files", tags=["files"])


class FileInfo(BaseModel):
    """Information about a whitelisted file."""
    path: str
    size: int
    content_type: str


@router.get("", response_model=List[FileInfo])
def list_files() -> List[FileInfo]:
    """
    List all files Sema is allowed to access.

    Returns a list of allowed files with their metadata:
    - path: Relative path from repo root
    - size: File size in bytes
    - content_type: "text", "pdf", "word", or "binary"

    Example response:
    ```json
    [
      {"path": "sem_mem/core.py", "size": 1234, "content_type": "text"},
      {"path": "docs/guide.pdf", "size": 5678, "content_type": "pdf"},
      ...
    ]
    ```
    """
    allowed = load_whitelist()
    files = get_file_info(allowed)
    return [FileInfo(**f) for f in files]


@router.get("/content")
def get_file(path: str = Query(..., description="Relative path to the file")):
    """
    Get the contents of a whitelisted file.

    - For text files: Returns plain text
    - For PDFs: Returns application/pdf bytes
    - For Word docs: Returns appropriate Word document bytes

    Args:
        path: Relative path as listed by GET /files

    Returns:
        File contents with appropriate content type

    Raises:
        404: File not found
        403: File not in whitelist (access denied)
    """
    allowed = load_whitelist()

    try:
        file_path = validate_file_access(path, allowed)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))

    content, media_type = read_file_content(file_path)

    if isinstance(content, bytes):
        return Response(content=content, media_type=media_type)
    else:
        return PlainTextResponse(content=content, media_type=media_type)


@router.get("/search")
def search_files(
    query: str = Query(..., min_length=1, description="Search query"),
    max_results: int = Query(20, ge=1, le=100, description="Maximum results to return"),
) -> List[dict]:
    """
    Search through whitelisted text files for matching content.

    Simple text search - returns files containing the query string.
    Only searches text files, not PDFs or Word docs.

    Args:
        query: Text to search for (case-insensitive)
        max_results: Maximum number of results to return

    Returns:
        List of matches with file path and matching line context
    """
    allowed = load_whitelist()
    files = get_file_info(allowed)

    results = []
    query_lower = query.lower()

    for file_info in files:
        if file_info["content_type"] != "text":
            continue

        try:
            file_path = validate_file_access(file_info["path"], allowed)
            content, _ = read_file_content(file_path)

            if not isinstance(content, str):
                continue

            # Find matching lines
            matches = []
            for line_num, line in enumerate(content.split("\n"), 1):
                if query_lower in line.lower():
                    matches.append({
                        "line": line_num,
                        "content": line.strip()[:200],  # Truncate long lines
                    })

                    if len(matches) >= 5:  # Max 5 matches per file
                        break

            if matches:
                results.append({
                    "path": file_info["path"],
                    "matches": matches,
                })

                if len(results) >= max_results:
                    break

        except (FileNotFoundError, PermissionError):
            continue

    return results
