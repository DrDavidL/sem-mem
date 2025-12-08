# Sem-Mem: Tiered Semantic Memory for AI Agents

> **Drop-in semantic memory for AI agents: local, HNSW-backed storage with auto-memory and RAG out of the box. Works with OpenAI, Claude, Gemini, Ollama, and more.**

## Who It's For

- **Personal assistant builders** who want their bot to remember user details across sessions
- **Domain-specific agents** (medical education, legal education, research) that need durable semantic memory
- **Developers wanting local control** over memory vs. remote vector databases

## Why Not Just Use a Hosted Vector DB?

You absolutely can (and should) use hosted vector databases for many workloads. Sem-Mem is for a different, narrower use case:

**When a hosted DB is great:**

- Large, multi-tenant applications
- High write concurrency from many clients
- Complex query patterns (filters, aggregations, hybrid search)
- Strict SLOs, replication, managed backups, etc.

**When Sem-Mem is a better fit:**

- **Personal or small-team assistants** where memory lives “with” the agent
- **Local-first workflows** where you don’t want another remote dependency
- **Lightweight deployment**: one Python process, no external services
- **Tight latency loops**: L1 cache + local HNSW is extremely fast for KNN

Sem-Mem’s design trades:

- **Horizontal scalability** and **multi-tenant complexity**

        for

- **Simplicity**, **local control**, and **low operational overhead**.

If you already run Postgres/pgvector, Qdrant, Pinecone, etc. and are happy with that, you may not need Sem-Mem. If you just want your agent to *remember things* without standing up new infra, Sem-Mem is likely a better fit.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  L1: SmartCache (RAM)                                       │
│  ├── Protected tier (frequently accessed)                   │
│  └── Probation tier (recently accessed)                     │
│ → HIT: Instant recall, no disk I/O                       │
└─────────────────────────┬───────────────────────────────────┘
                          │ MISS
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  L2: HNSW Index (Disk)                                      │
│ → O(log n) approximate nearest neighbor search             │
│ → Results promoted to L1 cache                             │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  LLM (OpenAI Responses API)                                 │
│ + Retrieved memories as context                            │
│ + Persistent instructions                                  │
│ + Optional web search                                      │
└─────────────────────────────────────────────────────────────┘
```
## Design Choices

Sem-Mem makes a few deliberate trade-offs:

### 1. Tiered Memory (L1 + L2)

- **L1 (SmartCache, RAM):** Segmented LRU with "Protected" and "Probation" tiers.
  - Recently / frequently accessed items stay hot.
  - Misses fall back to L2.
- **L2 (HNSW, Disk):** Persistent approximate nearest neighbor index.
  - All long-term memories live here.
  - Hits are promoted back into L1.

This mirrors how you probably want an assistant to behave: **fast recall for what’s active**, with the ability to rediscover older but relevant memories when needed.

### 2. Local Disk Instead of a Remote DB

- Uses a **local directory** (`MEMORY_STORAGE_DIR`) to store:
  - HNSW index files
  - Instructions file
- No separate DB service to deploy or manage.
- Fits well for:
  - Single-user setups
  - Small internal tools
  - Prototyping and research agents

You still send text to OpenAI for embeddings and chat, but **the semantic index itself stays on your machine**.

### 3. HNSW for Approximate Nearest Neighbor

- **Why HNSW?**
  - Excellent recall/latency trade-off for typical agent-sized memory (10³–10⁶ vectors).
  - Mature, well-tested implementation (`hnswlib`).
- Tuned with:
  - `M=16`, `ef_construction=200` by default (configurable in code).

The goal isn’t to be a general-purpose vector database, but a **fast, embeddable memory index** that feels “instant” in practice.

### 4. Memory-Aware Agent Abstractions

- `MemoryChat` and `@with_memory` / `@with_rag` decorators encapsulate:
  - Retrieval
  - Instruction injection
  - Optional web search
  - Auto-memory (salience detection)

Instead of wiring RAG + instructions + memory heuristics yourself every time, you call one object or wrap one function. Sem-Mem tries to be **just enough structure** to be useful, without locking you into a full agent framework.


## 90-Second Quickstart

```python
from sem_mem import SemanticMemory
from sem_mem.decorators import MemoryChat

memory = SemanticMemory(api_key="sk-...")
chat = MemoryChat(memory)

# The "aha" moment: it remembers you
print(chat.send("My name is Sam, I'm a cardiology fellow in Boston."))
print(chat.send("Where do I live and what do I do?"))
# → Model can answer from semantic memory, not just the last message

# Explicit memory storage
chat.remember("Patient prefers morning appointments")
```

### Medical Education Example

```python
# Store teaching case facts - (Not for clinical care - validation/security checks needed!)
```

## Features

* **Zero-Latency "Hot" Recall:** Uses a Segmented LRU cache to keep relevant context in RAM.
* **Local Storage:** Your memory index (HNSW) and instructions stay on your machine—not in a cloud database. Note: Text is still sent to OpenAI for embeddings and chat responses.
* **HNSW Index:** Approximate nearest neighbor search with O(log n) using hnswlib.
* **Query Expansion:** LLM-powered alternative query generation for better recall.
* **Auto-Memory (Salience Detection):** Automatically saves important exchanges (personal facts, decisions, corrections) without explicit user action.
* **Web Search:** Optional web search via OpenAI's Responses API (off by default).
* **Memory-Aware Context:** The model understands it has semantic memory and can help users manage it.
* **Thread-Safe:** Concurrent access support with RLock synchronization.
* **Model Selection:** Support for reasoning models (gpt-5.1, o1, o3) with configurable reasoning effort.
* **PDF Ingestion:** "Read" clinical guidelines or papers and auto-chunk them into long-term memory.
* **Memory Atlas:** A 2D visualization (PCA) of your knowledge graph to verify semantic clustering.
* **FastAPI Server:** RESTful API for programmatic access and microservice deployment.
* **Docker Support:** Containerized deployment with docker-compose.

## Installation

### From Source (Development)

```bash
git clone https://github.com/DrDavidL/sem-mem.git
cd sem-mem
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[all]"    # everything for local dev

# Core package only
pip install -e .

# With Streamlit app
pip install -e ".[app]"

# With FastAPI server
pip install -e ".[server]"

# Everything (app + server + dev tools)
pip install -e ".[all]"
```

### Package Installation

```bash
# Core package only
pip install sem-mem

# With Streamlit app
pip install "sem-mem[app]"

# With FastAPI server
pip install "sem-mem[server]"
```

## Configuration

Sem-Mem supports multiple configuration sources (in priority order):

1. **Environment variables** (highest priority)
2. **`.env` file** in project root
3. **Streamlit secrets** (`.streamlit/secrets.toml`)

### Required Configuration

Create a `.env` file in your project root:

```bash
# Copy from .env.example
cp .env.example .env
```

```env
# Required
OPENAI_API_KEY=sk-...

# Optional - Model Selection
CHAT_MODEL=gpt-5.1              # Options: gpt-5.1 (reasoning), gpt-4.1
REASONING_EFFORT=low            # For reasoning models: low, medium, high

# Optional - Storage
MEMORY_STORAGE_DIR=./local_memory
CACHE_SIZE=20
EMBEDDING_MODEL=text-embedding-3-small

# Optional - API Server
SEMMEM_API_URL=http://localhost:8000
```

## Multi-Provider Support

Sem-Mem supports multiple LLM providers for chat and embeddings. You can mix providers (e.g., Claude for chat, OpenAI for embeddings).

### Available Providers

| Provider | Chat | Embeddings | Tier | Notes |
|----------|------|------------|------|-------|
| **OpenAI** | ✅ | ✅ | 1 | Default, full feature support |
| **Azure OpenAI** | ✅ | ✅ | 1 | Same SDK, deployment names |
| **Anthropic (Claude)** | ✅ | ❌ | 2 | Pair with OpenAI/Google for embeddings |
| **Google (Gemini)** | ✅ | ✅ | 2 | text-embedding-004 (768d) |
| **Ollama** | ✅ | ✅ | 2 | Local inference |
| **OpenRouter** | ✅ | ❌ | 3 | Multi-model gateway |

### Configuration

Set provider via environment variables:

```bash
# Chat provider (can be changed anytime)
SEMMEM_CHAT_PROVIDER=openai   # openai, azure, anthropic, google, ollama, openrouter

# Embedding provider (locked after first use!)
SEMMEM_EMBEDDING_PROVIDER=openai   # openai, azure, google, ollama
```

Each provider has its own API key env var. See [.env.example](.env.example) for the full list.

### Example: Claude for Chat, OpenAI for Embeddings

```bash
SEMMEM_CHAT_PROVIDER=anthropic
SEMMEM_CHAT_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...

SEMMEM_EMBEDDING_PROVIDER=openai
SEMMEM_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...
```

Or in code:

```python
from sem_mem import SemanticMemory

memory = SemanticMemory(
    chat_provider="anthropic",
    embedding_provider="openai",
)
```

### Installation

```bash
# Core (OpenAI + Azure)
pip install sem-mem

# With specific providers
pip install "sem-mem[anthropic]"   # Anthropic/Claude
pip install "sem-mem[google]"      # Google/Gemini
pip install "sem-mem[all-providers]"  # All providers
```

### Embedding Provider Lock

**Important:** Once you create an index with a specific embedding provider/model, it cannot be changed without deleting or migrating the index. This is because different embedding models produce vectors with different dimensions and semantics.

If you try to use a different embedding provider with an existing index, you'll get an `EmbeddingMismatchError` with instructions for how to proceed.

## Usage

### Option 1: Streamlit App (Standalone)

The simplest way to use Sem-Mem with a full UI:

```bash
streamlit run app.py
```

### Option 2: FastAPI Server + API Client

For microservice deployment or programmatic access:

```bash
# Terminal 1: Start the API server
uvicorn server:app --reload

# Terminal 2: Run the API client UI
streamlit run app_api.py
```

### Option 3: Docker Deployment

```bash
# Build and run both services
docker-compose up --build

# Access:
# - API Server: http://localhost:8000
# - Streamlit UI: http://localhost:8501
```

### Option 4: Python API

#### MemoryChat Class (Recommended)

```python
from sem_mem import SemanticMemory
from sem_mem.decorators import MemoryChat

memory = SemanticMemory(api_key="sk-...")
chat = MemoryChat(memory)

# Stateful conversation with automatic RAG
response = chat.send("Hello, I'm a physician.")
response = chat.send("What's my profession?")  # Remembers context

# Memory operations
chat.remember("Patient prefers morning appointments")
chat.add_instruction("Always be concise")
chat.save_thread()  # Save conversation to L2

chat.new_thread()  # Fresh conversation, same memory
```

#### Decorators

```python
from sem_mem import SemanticMemory, with_memory, with_rag

memory = SemanticMemory(api_key="sk-...")

# Full memory integration
@with_memory(memory)
def chat(user_input: str, context: str = "", instructions: str = "", **_) -> str:
    # context = retrieved memories, instructions = from instructions.txt
    return my_llm(f"{instructions}\n\n{context}\n\nUser: {user_input}")

# RAG only (just retrieval)
@with_rag(memory)
def simple_chat(user_input: str, context: str = "", **_) -> str:
    return my_llm(f"{context}\n\n{user_input}")
```

#### Direct API

```python
from sem_mem import SemanticMemory

memory = SemanticMemory(
    api_key="sk-...",
    chat_model="gpt-5.1",           # or "gpt-4.1"
    reasoning_effort="medium",       # for reasoning models
    auto_memory=True,                # auto-save important exchanges (default)
    web_search=False,                # enable web search (default: off)
)

# Store facts
memory.remember("The patient is allergic to penicillin")
memory.add_instruction("I am an informatics physician")

# Query with RAG
response, resp_id, mems, logs = memory.chat_with_memory(
    "What allergies should I check?",
    previous_response_id=prev_id,  # For conversation continuity
    web_search=True,               # Enable web search for this query
)
```

## Model Selection

Sem-Mem supports OpenAI models via the `chat_model` parameter:

| Model | Type | Features |
|-------|------|----------|
| `o3` | Reasoning | Extended thinking, `reasoning_effort` parameter |
| `o1` | Reasoning | Extended thinking, `reasoning_effort` parameter |
| `gpt-4o` | Standard | Temperature control, faster responses |
| `gpt-4.1` | Standard | Temperature control, optimized for code |

> **Note**: The codebase uses `gpt-5.1` as a placeholder for reasoning models. Pass any valid OpenAI model string (e.g., `o3`, `o1`, `gpt-4o`) via `chat_model`.

### Reasoning Effort (for o1, o3, and other reasoning models)

- **low**: Quick responses, minimal reasoning
- **medium**: Balanced reasoning depth
- **high**: Thorough analysis, slower but more accurate

Change models via:
- **UI**: Sidebar model selector
- **API**: `PUT /model?model=gpt-4o`
- **Config**: `CHAT_MODEL=o3` in `.env`
- **Code**: `SemanticMemory(chat_model="o3")`

## Web Search

Sem-Mem optionally integrates OpenAI's built-in web search tool for real-time information retrieval. **Web search is off by default.**

> **Clarification**: "Local storage" refers to your HNSW index and instructions file staying on your machine. When web search is enabled, queries are sent to OpenAI's web search API (in addition to the normal embedding/chat API calls).

Enable via:
- **UI**: Toggle "🌐 Web Search" in the sidebar
- **Code**: `SemanticMemory(web_search=True)` or `chat_with_memory(..., web_search=True)`
- **API**: `POST /chat` with `{"query": "...", "web_search": true}`

When enabled, the model can search the web to answer questions about current events, recent data, or topics not in your semantic memory.

## Chat Commands

In the Streamlit chat interface:

- **Regular text**: Query with RAG from semantic memory
- `remember: <text>`: Add fact to semantic memory (L2)
- `instruct: <text>`: Add permanent instruction (persisted to instructions.txt)

## API Endpoints

When running the FastAPI server:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Chat with RAG (supports `model`, `reasoning_effort` params) |
| `/remember` | POST | Store a single memory |
| `/remember/batch` | POST | Store multiple memories |
| `/recall` | POST | Retrieve relevant memories |
| `/instructions` | GET/PUT/POST | Manage system instructions |
| `/cache` | GET | View L1 cache state |
| `/stats` | GET | Memory statistics |
| `/model` | GET/PUT | View/update model configuration |
| `/upload/pdf` | POST | Ingest PDF document |
| `/threads/save` | POST | Save conversation to L2 |

API documentation available at `http://localhost:8000/docs` when server is running.

## Project Structure

```text
.
├── sem_mem/
│   ├── __init__.py          # Package exports
│   ├── core.py              # SemanticMemory, SmartCache, MEMORY_SYSTEM_CONTEXT
│   ├── async_core.py        # Async version for FastAPI
│   ├── config.py            # Configuration and secret loading
│   ├── vector_index.py      # HNSWIndex for L2 storage
│   ├── decorators.py        # @with_memory, @with_rag, MemoryChat
│   ├── auto_memory.py       # AutoMemory salience detection
│   └── thread_utils.py      # Thread utilities (title generation, summarization)
├── tests/                   # Unit tests
├── local_memory/            # HNSW index files + instructions.txt
├── app.py                   # Streamlit standalone app
├── app_api.py               # Streamlit API client app
├── server.py                # FastAPI server
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Thread Management

Sem-Mem provides optional thread utilities that UIs can use to enhance conversation management. The thread model is **intentionally UI-layer**, not baked into core classes like `MemoryChat`.

### Auto-Rename Threads

Threads are automatically given descriptive titles after enough context accumulates:

```python
from sem_mem import generate_thread_title

messages = [
    {"role": "user", "content": "How do I configure nginx for SSL?"},
    {"role": "assistant", "content": "Here's how to set up SSL..."},
    {"role": "user", "content": "What about certificate renewal?"},
    {"role": "assistant", "content": "For automatic renewal, use certbot..."},
]

title = generate_thread_title(messages)  # "Nginx SSL Configuration and Renewal"
```

**Configuration** (in `sem_mem/config.py` or environment):

| Setting | Default | Description |
|---------|---------|-------------|
| `AUTO_THREAD_RENAME_ENABLED` | `True` | Enable auto-rename feature |
| `AUTO_THREAD_RENAME_MIN_USER_MESSAGES` | `3` | Minimum user messages before renaming |
| `AUTO_THREAD_RENAME_MODEL` | `gpt-4.1-mini` | Model for title generation |
| `AUTO_THREAD_RENAME_MAX_WORDS` | `8` | Maximum words in title |

### Conversation Summaries (Windowed)

Long conversations are automatically summarized into "chapters" stored in L2. This creates durable semantic history focusing on:
- Key decisions and conclusions
- User preferences and constraints
- Corrections and clarifications
- Important background context

```python
from sem_mem import estimate_message_tokens, select_summary_window, summarize_conversation_window

messages = thread["messages"]
windows = thread.get("summary_windows", [])

# Check if summarization is needed
if estimate_message_tokens(messages) >= 8000:
    window = select_summary_window(messages, windows, leave_recent=6)
    if window:
        start, end = window
        summary = summarize_conversation_window(messages[start:end], client=memory.client)
        # Store summary in L2 and track in thread
```

**Configuration** (in `sem_mem/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `CONVERSATION_SUMMARY_TOKEN_THRESHOLD` | `8000` | Tokens before summarizing |
| `CONVERSATION_SUMMARY_MIN_MESSAGES` | `10` | Minimum messages to consider |
| `CONVERSATION_SUMMARY_MODEL` | `gpt-4.1-mini` | Model for summarization |
| `CONVERSATION_SUMMARY_LEAVE_RECENT` | `6` | Recent messages to keep raw |
| `CONVERSATION_SUMMARY_MAX_WINDOWS_PER_THREAD` | `10` | Max summaries per thread |

Summaries are stored in L2 with metadata linking them to the source thread and message range, enabling semantic retrieval of historical conversation context.

### Thread Deletion with Farewell Summary

When deleting a thread, you can optionally save a "farewell summary" to L2 memory before removal. This preserves durable takeaways (decisions, preferences, important context) even after the thread is gone.

```python
from sem_mem import summarize_deleted_thread

messages = thread["messages"]
summary = summarize_deleted_thread(messages, client=memory.client)
if summary:
    memory.remember(summary, metadata={"type": "farewell_summary", "thread_name": name})
```

**Configuration** (in `sem_mem/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `ON_DELETE_THREAD_BEHAVIOR` | `"prompt"` | `"prompt"`, `"always_save"`, or `"never_save"` |
| `FAREWELL_SUMMARY_MODEL` | `gpt-4.1-mini` | Model for farewell summarization |
| `FAREWELL_SUMMARY_MAX_CHARS` | `8000` | Max chars of conversation to include |

**Behavior options:**
- `"prompt"`: Ask user whether to save summary before deleting (default)
- `"always_save"`: Always generate and save farewell summary
- `"never_save"`: Delete immediately without summarization

The farewell summary focuses on the same durable information as windowed summaries: key decisions, user preferences, corrections, and important context.

### Thread Data Structure (UI Layer)

The Streamlit app uses this structure for threads:

```python
{
    "messages": [],              # List of {role, content} dicts
    "response_id": None,         # OpenAI Responses API ID for continuity
    "title": "New conversation", # Display title
    "title_user_overridden": False,  # If True, auto-rename won't touch it
    "summary_windows": [         # Summaries of earlier conversation segments
        {
            "start_index": 0,
            "end_index": 24,
            "summary_text": "...",
            "summary_id": "...",   # L2 memory ID
            "timestamp": "...",
        }
    ],
}
```

**Why threads live in the UI layer:**
- Different surfaces (Streamlit, FastAPI, CLI) can define their own thread models
- `MemoryChat` stays simple: just `messages` + `previous_response_id`
- Thread utilities are pure functions that any UI can call

### Building Your Own Thread UI

If you're building a custom frontend:

```python
from sem_mem import SemanticMemory, generate_thread_title
from sem_mem.config import AUTO_THREAD_RENAME_MIN_USER_MESSAGES

memory = SemanticMemory(api_key="...")

# Your thread storage (DB, session state, etc.)
thread = {
    "messages": [],
    "response_id": None,
    "title": "New conversation",
    "title_user_overridden": False,
}

# After each chat turn
response, resp_id, mems, logs = memory.chat_with_memory(
    user_input, previous_response_id=thread["response_id"]
)
thread["response_id"] = resp_id
thread["messages"].append({"role": "user", "content": user_input})
thread["messages"].append({"role": "assistant", "content": response})

# Auto-rename check
user_count = sum(1 for m in thread["messages"] if m["role"] == "user")
if (
    not thread["title_user_overridden"]
    and thread["title"] == "New conversation"
    and user_count >= AUTO_THREAD_RENAME_MIN_USER_MESSAGES
):
    thread["title"] = generate_thread_title(thread["messages"], client=memory.client)
```

## Memory Systems

| Memory Type | Storage | Scope | Eviction | User Update | Auto Update |
|-------------|---------|-------|----------|-------------|-------------|
| **Instructions** | `instructions.txt` | All threads | Never | Sidebar editor, `instruct:` cmd | Never |
| **L1 (Hot Cache)** | RAM (SmartCache) | All threads | LRU | None (automatic) | Promoted from L2 on access |
| **L2 (Cold Storage)** | HNSW index | All threads | Never | `remember:` cmd, PDF upload | Auto-saved from L1 after 6+ hits; Auto-memory |
| **Thread History** | Session state | Per thread | New thread | None | Automatic on chat |

### Auto-Memory (Salience Detection)

Auto-memory automatically saves important exchanges without explicit user action. Enabled by default for API usage.

**What gets auto-saved:**
- Personal facts: "I'm a physician", "My name is David"
- Explicit markers: "Remember that...", "This is important"
- Decisions/conclusions reached in conversation
- Corrections to previous assumptions

**How it works:**
1. Cheap heuristics check for obvious signals (no API cost)
2. For ambiguous cases, uses `gpt-4.1-mini` to evaluate salience
3. High-salience content is distilled and saved to L2

```python
# Enabled by default
memory = SemanticMemory(api_key="...")

# Disable globally
memory = SemanticMemory(api_key="...", auto_memory=False)

# Disable per-call
response = memory.chat_with_memory(query, auto_remember=False)
```

### Memory Policy

Sema uses a policy-based approach to decide what to remember:

**Prioritized for storage:**
- Stable identity facts (roles, background, long-term projects)
- Explicit preferences ("I prefer concise answers", "Please always...")
- Decisions and plans agreed on together
- Corrections and updates to previous information

**Treated cautiously:**
- Momentary moods or transient states
- Sensitive details (unless explicitly requested)
- Routine Q&A without durable value

**Correction handling:**
When a user corrects or updates something, the new information overrides the old:
- Corrections are tagged with `kind: "correction"` in metadata
- On retrieval, corrections appear last in the context (so the model sees them as most recent)
- The model is instructed to treat corrections as overriding older facts

```python
# Auto-detected corrections get special treatment
"Actually, I changed my mind about the architecture..."  # Tagged as correction
"No, that's not right - it should be..."                 # Tagged as correction
"I prefer X going forward..."                            # Tagged as preference
```

**Structured memory format:**
Auto-saved memories include metadata for smarter retrieval:
```json
{
  "kind": "correction",
  "text": "Actually, I prefer dark mode",
  "timestamp": "2025-01-15T10:30:00",
  "meta": {"source": "auto_memory", "salience": 0.85}
}
```

### Multi-Session Behavior

**Standalone Streamlit (`app.py`):**
- **L2 (Disk)**: Shared across all sessions - memories persist and are accessible to everyone
- **L1 (RAM)**: Isolated per session - each browser tab has its own cache
- Best for single-user or development use

**FastAPI Server (`server.py` + `app_api.py`):**
- **Both L1 and L2**: Fully shared across all connected clients
- Single memory instance serves all API requests
- Best for multi-user production deployments

For shared memory across multiple users, use the API architecture:
```bash
uvicorn server:app --reload    # Shared backend
streamlit run app_api.py       # Multiple clients can connect
```

## Migration from LSH

If you have existing data in the old LSH bucket format (`bucket_*.json`), you can migrate to HNSW:

```python
from sem_mem import migrate_lsh_to_hnsw

# Migrate existing buckets to HNSW index
count = migrate_lsh_to_hnsw("./local_memory")
print(f"Migrated {count} memories to HNSW index")
```

## Development

```bash
# Setup
git clone https://github.com/DrDavidL/sem-mem.git
cd sem-mem
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run tests
pytest

# Lint
ruff check .
```

<img width="2242" height="1672" alt="CleanShot 2025-12-06 at 18 33 59@2x" src="https://github.com/user-attachments/assets/5a81b41f-d082-4cd5-a4bf-3360c4d7a529" />

## Limitations

Sem-Mem is intentionally small and local. Some things it does **not** try to do:

- **Not a general-purpose vector database.**  
  No sharding, distributed indexing, replica sets, or advanced filtering/analytics. If you need that, use pgvector, Qdrant, Pinecone, etc.

- **Single-node, local storage only.**  
  All memory lives in a local directory (`MEMORY_STORAGE_DIR`). There’s no built-in remote persistence, multi-region replication, or cross-node synchronization.

- **Best for 10³–10⁶ memories, not billions.**  
  HNSW can handle large indexes, but Sem-Mem isn’t tuned for multi‑billion‑vector workloads.

- **Coarse-grained access control.**  
  The FastAPI server shares one memory instance across clients. There is no built-in per-user auth/namespace isolation yet.

- **Eventual consistency for auto-memory.**  
  Auto-memory is opportunistic: important facts are distilled and saved, but not every utterance is guaranteed to be captured or perfectly summarized.

- **LLM-dependent behavior.**  
  Query expansion, auto-memory salience detection, and chat behavior depend on the underlying OpenAI models. Changes in model behavior can change Sem-Mem’s “feel” without any code changes.

## Technical Details

- **HNSW**: Hierarchical Navigable Small World graph for O(log n) approximate nearest neighbor search
- **hnswlib**: Lightweight C++ library with Python bindings (ef_construction=200, M=16)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536 dimensions, configurable)
- **Query Expansion**: Uses `gpt-4.1-mini` to generate alternative query phrasings
- **Cache**: Segmented LRU with Protected/Probation tiers
- **API**: OpenAI Responses API for stateful conversations
- **Thread Safety**: RLock synchronization for concurrent access

## Production Notes

Sem-Mem can be used in small production setups, but keep these in mind:

- **Single process = single point of failure.**  
  The default FastAPI + `SemanticMemory` instance is in-memory for L1 and on local disk for L2. If the process dies, L1 is lost (L2 persists), and you need a restart mechanism (systemd, Docker, Kubernetes, etc.).

- **Concurrency model.**  
  Thread safety is handled with `RLock`, but:
  - One Python process still has the GIL.
  - Very high concurrency (hundreds of RPS) may require:
    - Multiple worker processes
    - One shared disk index mounted read/write
    - Or per-worker copies with a periodic sync strategy

- **Backups and durability.**  
  The HNSW index and instructions are just files:
  - Use regular filesystem backups / snapshots if memories are important.
  - Consider versioning the memory directory for “time travel” or rollback.

- **Latency vs. index size.**  
  As L2 grows, you may want to:
  - Tune HNSW parameters (`ef`, `M`) for your workload
  - Prune or archive very old/low-value memories
  - Increase RAM if you expect a very large hot-set in L1

- **Per-user isolation (multi-tenant apps).**  
  For real multi-user production:
  - Run **separate memory directories per user** (e.g., `./memory/{user_id}`), or
  - Add a user/tenant dimension at the API layer and instantiate `SemanticMemory` per tenant.
  - True multi-tenant sharing with fine-grained ACLs is **out of scope** for now.

- **Observability.**  
  Sem-Mem exposes `/stats` and `/cache` endpoints, but:
  - You should still add your own logging/tracing around API calls.
  - In latency-sensitive contexts, measure:
    - Cache hit rate (L1 vs L2)
    - HNSW query times
    - End-to-end request latency

In short: Sem-Mem is great for **local-first assistants, internal tools, and research agents**. If you’re building a large, multi-tenant SaaS with strict SLOs, you’ll likely pair Sem-Mem with more traditional infrastructure (or use a hosted vector DB directly).

## License

MIT License
