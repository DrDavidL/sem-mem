"""
FastAPI server for Sem-Mem semantic memory system.

Run with: uvicorn server:app --reload
"""

from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sem_mem.async_core import AsyncSemanticMemory
from sem_mem.config import get_api_key, get_config, CHAT_MODELS, REASONING_EFFORTS, DEFAULT_CHAT_MODEL


# --- Pydantic Models ---

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    previous_response_id: Optional[str] = Field(None, description="For conversation continuity")
    model: Optional[str] = Field(None, description="Chat model (gpt-5.1, gpt-4.1)")
    reasoning_effort: Optional[str] = Field(None, description="For reasoning models: low, medium, high")


class ChatResponse(BaseModel):
    response: str
    response_id: str
    memories: List[str]
    logs: List[str]


class RememberRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to remember")
    metadata: Optional[dict] = None


class RememberBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="Texts to remember")
    metadata: Optional[dict] = None


class InstructionRequest(BaseModel):
    instruction: str = Field(..., min_length=1)


class RecallRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(3, ge=1, le=10)
    threshold: float = Field(0.40, ge=0.0, le=1.0)
    expand_query: bool = Field(True, description="Use LLM to generate alternative query phrasings")


class RecallResponse(BaseModel):
    memories: List[str]
    logs: List[str]


class MemoryStats(BaseModel):
    total_l2_memories: int
    l1_cache_size: int
    instructions_length: int
    chat_model: str
    reasoning_effort: str


class ModelConfig(BaseModel):
    available_models: List[str]
    current_model: str
    reasoning_efforts: List[str]
    current_reasoning_effort: str
    is_reasoning_model: bool


class CacheItem(BaseModel):
    text: str
    tier: str  # "protected" or "probation"


class CacheState(BaseModel):
    protected: List[str]
    probation: List[str]


# --- App Setup ---

memory: Optional[AsyncSemanticMemory] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize memory on startup."""
    global memory
    config = get_config()
    api_key = config["api_key"]

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it via:\n"
            "  - Environment variable: export OPENAI_API_KEY=sk-...\n"
            "  - .env file: OPENAI_API_KEY=sk-...\n"
            "  - .streamlit/secrets.toml: OPENAI_API_KEY=\"sk-...\""
        )

    memory = AsyncSemanticMemory(
        api_key=api_key,
        storage_dir=config["storage_dir"],
        cache_size=config["cache_size"],
        embedding_model=config["embedding_model"],
        chat_model=config["chat_model"],
        reasoning_effort=config["reasoning_effort"],
    )
    yield
    # Cleanup if needed


app = FastAPI(
    title="Sem-Mem API",
    description="Tiered semantic memory system for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_memory() -> AsyncSemanticMemory:
    """Dependency to get memory instance."""
    if memory is None:
        raise HTTPException(status_code=503, detail="Memory not initialized")
    return memory


# --- Endpoints ---

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "memory_initialized": memory is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """
    Chat with RAG-enhanced responses.
    Maintains conversation state via previous_response_id.
    Optionally specify model and reasoning_effort.
    """
    response_text, response_id, memories, logs = await mem.chat_with_memory(
        request.query,
        previous_response_id=request.previous_response_id,
        model=request.model,
        reasoning_effort=request.reasoning_effort,
    )
    return ChatResponse(
        response=response_text,
        response_id=response_id,
        memories=memories,
        logs=logs
    )


@app.post("/remember")
async def remember(
    request: RememberRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Store a single memory."""
    msg = await mem.remember(request.text, request.metadata)
    return {"message": msg}


@app.post("/remember/batch")
async def remember_batch(
    request: RememberBatchRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Store multiple memories concurrently."""
    count = await mem.remember_batch(request.texts, request.metadata)
    return {"added": count, "total_submitted": len(request.texts)}


@app.post("/recall", response_model=RecallResponse)
async def recall(
    request: RecallRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Retrieve relevant memories without chat."""
    memories, logs = await mem.recall(
        request.query,
        limit=request.limit,
        threshold=request.threshold,
        expand_query=request.expand_query,
    )
    return RecallResponse(memories=memories, logs=logs)


@app.get("/instructions")
async def get_instructions(mem: AsyncSemanticMemory = Depends(get_memory)):
    """Get current instructions."""
    text = await mem.load_instructions()
    return {"instructions": text}


@app.put("/instructions")
async def set_instructions(
    request: InstructionRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Replace all instructions."""
    await mem.save_instructions(request.instruction)
    return {"message": "Instructions saved"}


@app.post("/instructions")
async def add_instruction(
    request: InstructionRequest,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Append an instruction."""
    await mem.add_instruction(request.instruction)
    return {"message": "Instruction added"}


@app.get("/cache", response_model=CacheState)
async def get_cache(mem: AsyncSemanticMemory = Depends(get_memory)):
    """Get current L1 cache state."""
    items = mem.local_cache.list_items()
    return CacheState(
        protected=[i['text'][:100] for i in items["Protected (Sticky)"]],
        probation=[i['text'][:100] for i in items["Probation (Transient)"]]
    )


@app.get("/stats", response_model=MemoryStats)
async def get_stats(mem: AsyncSemanticMemory = Depends(get_memory)):
    """Get memory statistics."""
    stats = mem.get_stats()
    instructions = await mem.load_instructions()

    return MemoryStats(
        total_l2_memories=stats["l2_memories"],
        l1_cache_size=stats["l1_cache_size"],
        instructions_length=len(instructions),
        chat_model=mem.chat_model,
        reasoning_effort=mem.reasoning_effort,
    )


@app.get("/model", response_model=ModelConfig)
async def get_model_config(mem: AsyncSemanticMemory = Depends(get_memory)):
    """Get current model configuration."""
    return ModelConfig(
        available_models=list(CHAT_MODELS.keys()),
        current_model=mem.chat_model,
        reasoning_efforts=REASONING_EFFORTS,
        current_reasoning_effort=mem.reasoning_effort,
        is_reasoning_model=mem.chat_model in ("gpt-5.1", "o1", "o3"),
    )


@app.put("/model")
async def set_model(
    model: str,
    reasoning_effort: Optional[str] = None,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Update the chat model and reasoning effort."""
    if model not in CHAT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {list(CHAT_MODELS.keys())}"
        )
    if reasoning_effort and reasoning_effort not in REASONING_EFFORTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reasoning_effort. Choose from: {REASONING_EFFORTS}"
        )

    mem.chat_model = model
    if reasoning_effort:
        mem.reasoning_effort = reasoning_effort

    return {
        "model": mem.chat_model,
        "reasoning_effort": mem.reasoning_effort,
        "is_reasoning_model": model in ("gpt-5.1", "o1", "o3"),
    }


@app.post("/upload/pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    chunk_size: int = 500,
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Upload and process a PDF file."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    from pypdf import PdfReader
    import io

    content = await file.read()
    reader = PdfReader(io.BytesIO(content))

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Chunk the text
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 20:
            chunks.append(chunk)

    # Store concurrently
    count = await mem.bulk_learn_texts(chunks, metadata={"source": "pdf", "filename": file.filename})

    return {
        "filename": file.filename,
        "pages": len(reader.pages),
        "chunks_created": len(chunks),
        "chunks_stored": count
    }


@app.post("/threads/save")
async def save_thread(
    messages: List[dict],
    thread_name: str = "api_thread",
    mem: AsyncSemanticMemory = Depends(get_memory)
):
    """Save a conversation thread to L2 memory."""
    if not messages:
        return {"chunks_saved": 0}

    # Build conversation text
    conversation_parts = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conversation_parts.append(f"{role.upper()}: {content}")

    full_conversation = "\n\n".join(conversation_parts)

    # Chunk long conversations
    chunk_size = 1500
    chunks = []
    if len(full_conversation) <= chunk_size:
        chunks.append(full_conversation)
    else:
        current_chunk = ""
        for part in conversation_parts:
            if len(current_chunk) + len(part) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = part
            else:
                current_chunk += "\n\n" + part if current_chunk else part
        if current_chunk:
            chunks.append(current_chunk.strip())

    count = await mem.bulk_learn_texts(
        chunks,
        metadata={"source": "thread", "thread_name": thread_name}
    )

    return {"chunks_saved": count}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
