"""
FastAPI router for backup and thread persistence endpoints.

Provides endpoints for:
- Exporting memory data
- Creating and restoring backups
- Managing persistent conversation threads
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field


# =============================================================================
# Request/Response Models
# =============================================================================


class BackupStats(BaseModel):
    """Statistics from a backup operation."""
    memory_count: int
    thread_count: int
    created_at: str


class BackupCreateResponse(BaseModel):
    """Response from creating a backup."""
    path: str
    stats: BackupStats


class BackupInfo(BaseModel):
    """Information about an available backup."""
    name: str
    path: str
    created_at: str
    memory_count: int
    thread_count: int
    version: str = "1.0"


class RestoreRequest(BaseModel):
    """Request body for restore operation."""
    backup_name: str = Field(..., description="Backup filename (with or without .json)")
    merge: bool = Field(False, description="Merge with existing data instead of replacing")
    re_embed: bool = Field(False, description="Recompute embeddings from text")


class RestoreResponse(BaseModel):
    """Response from restore operation."""
    memories_added: int
    memories_skipped: int
    threads_restored: int
    re_embedded: int
    instructions_action: str  # "replaced" or "kept"


class ThreadsResponse(BaseModel):
    """Response containing all threads."""
    version: str = "1.0"
    threads: Dict[str, Dict]


class ThreadData(BaseModel):
    """Thread data structure."""
    messages: List[Dict] = Field(default_factory=list)
    response_id: Optional[str] = None
    title: str = "New conversation"
    title_user_overridden: bool = False
    summary_windows: List[Dict] = Field(default_factory=list)


class SaveThreadsRequest(BaseModel):
    """Request body for saving all threads."""
    threads: Dict[str, Dict]


# =============================================================================
# Dependency to get memory instance
# =============================================================================


def get_memory():
    """
    Dependency to get the SemanticMemory or AsyncSemanticMemory instance.

    This should be overridden by the application to provide the actual
    memory instance. Example:

        from sem_mem import SemanticMemory
        from sem_mem.api.backup import backup_router, threads_router

        memory = SemanticMemory(api_key="...")

        def get_memory_override():
            return memory

        app.dependency_overrides[get_memory] = get_memory_override
        app.include_router(backup_router)
        app.include_router(threads_router)
    """
    raise HTTPException(
        status_code=500,
        detail="Memory instance not configured. Override get_memory dependency."
    )


# =============================================================================
# Backup Router
# =============================================================================


backup_router = APIRouter(prefix="/backup", tags=["backup"])


@backup_router.get("/export")
async def export_data(
    include_vectors: bool = Query(True, description="Include embedding vectors"),
    include_threads: bool = Query(True, description="Include conversation threads"),
    memory=Depends(get_memory),
) -> Dict:
    """
    Export all memory data as JSON.

    Returns the complete backup data in-memory (no file created).
    Use POST /backup/create to save to a file.
    """
    # export_all is sync in AsyncSemanticMemory
    return memory.export_all(
        include_vectors=include_vectors,
        include_threads=include_threads,
    )


@backup_router.post("/create", response_model=BackupCreateResponse)
async def create_backup(
    backup_name: Optional[str] = Query(
        None,
        description="Custom backup name (without .json). Uses timestamp if not provided."
    ),
    memory=Depends(get_memory),
) -> BackupCreateResponse:
    """
    Create a timestamped backup file on disk.

    Returns the relative path to the backup file and statistics.
    Note: Only relative paths are returned for security.
    """
    result = await memory.backup(backup_name=backup_name)
    return BackupCreateResponse(
        path=result["path"],
        stats=BackupStats(**result["stats"]),
    )


@backup_router.get("/list", response_model=List[BackupInfo])
async def list_backups(memory=Depends(get_memory)) -> List[BackupInfo]:
    """
    List available backups.

    Returns metadata about each backup including name, path, creation time,
    and counts of memories and threads.
    """
    # list_backups is sync in AsyncSemanticMemory
    backups = memory.list_backups()
    return [BackupInfo(**b) for b in backups]


@backup_router.post("/restore", response_model=RestoreResponse)
async def restore_backup(
    request: RestoreRequest = Body(...),
    memory=Depends(get_memory),
) -> RestoreResponse:
    """
    Restore from a backup file.

    Args:
        backup_name: Backup filename (looked up in backups directory)
        merge: If true, merge with existing data. If false, replace all.
        re_embed: If true, recompute vectors using current embedding model.

    Returns statistics about the restore operation.
    """
    try:
        result = await memory.restore(
            backup_name=request.backup_name,
            merge=request.merge,
            re_embed=request.re_embed,
        )
        return RestoreResponse(**result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Backup not found: {request.backup_name}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


@backup_router.delete("/{backup_name}")
async def delete_backup(
    backup_name: str,
    memory=Depends(get_memory),
) -> Dict:
    """
    Delete a backup file.

    Args:
        backup_name: Backup filename (with or without .json)

    Returns {"deleted": true} if successful.
    """
    # delete_backup is sync in AsyncSemanticMemory
    if memory.delete_backup(backup_name):
        return {"deleted": True, "name": backup_name}
    raise HTTPException(
        status_code=404,
        detail=f"Backup not found: {backup_name}"
    )


# =============================================================================
# Threads Router
# =============================================================================


threads_router = APIRouter(prefix="/threads", tags=["threads"])


@threads_router.get("", response_model=ThreadsResponse)
async def get_threads(memory=Depends(get_memory)) -> ThreadsResponse:
    """
    Get all persisted conversation threads.

    Returns all threads in the global namespace.
    Note: Thread namespace is global; user-scoping can be added later.
    """
    threads = await memory.load_threads()
    return ThreadsResponse(threads=threads)


@threads_router.get("/{thread_name}")
async def get_thread(
    thread_name: str,
    memory=Depends(get_memory),
) -> Dict:
    """
    Get a single thread by name.

    Returns the thread data if found.
    """
    thread = await memory.get_thread(thread_name)
    if thread is None:
        raise HTTPException(
            status_code=404,
            detail=f"Thread not found: {thread_name}"
        )
    return thread


@threads_router.post("")
async def save_threads(
    request: SaveThreadsRequest = Body(...),
    memory=Depends(get_memory),
) -> Dict:
    """
    Save all threads to persistent storage.

    Replaces all existing threads with the provided data.
    """
    await memory.save_threads(request.threads)
    return {
        "saved": True,
        "thread_count": len(request.threads),
    }


@threads_router.put("/{thread_name}")
async def save_thread(
    thread_name: str,
    thread: Dict = Body(...),
    memory=Depends(get_memory),
) -> Dict:
    """
    Save or update a single thread.

    Creates the thread if it doesn't exist, or updates if it does.
    """
    await memory.save_thread(thread_name, thread)
    return {
        "saved": True,
        "name": thread_name,
    }


@threads_router.delete("/{thread_name}")
async def delete_thread(
    thread_name: str,
    memory=Depends(get_memory),
) -> Dict:
    """
    Delete a thread from persistent storage.

    Returns {"deleted": true} if successful.
    """
    if await memory.delete_thread(thread_name):
        return {"deleted": True, "name": thread_name}
    raise HTTPException(
        status_code=404,
        detail=f"Thread not found: {thread_name}"
    )
