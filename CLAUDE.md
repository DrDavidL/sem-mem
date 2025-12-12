# Sem-Mem Project

Tiered semantic memory system for AI agents using HNSW-based vector indexing.

## Architecture

- **SmartCache (L1)**: Segmented LRU in RAM with Protected/Probation tiers
- **L2 Storage**: HNSW index (`hnsw_index.bin` + `hnsw_metadata.json`) for O(log n) semantic search
- **Instructions**: Persistent system instructions in `local_memory/instructions.txt`
- **UI**: Streamlit app (`app.py`)

## Key Files

- `sem_mem/core.py` - SmartCache, SemanticMemory classes
- `sem_mem/vector_index.py` - HNSWIndex class for L2 storage
- `app.py` - Streamlit frontend
- `local_memory/` - HNSW index files + instructions.txt

## Memory Systems

| Memory Type | Storage | Scope | Eviction | User Update | Auto Update |
|-------------|---------|-------|----------|-------------|-------------|
| **Global Instructions** | `instructions.txt` | All threads | Never | Sidebar editor, `instruct:` cmd, file edit | Never |
| **Thread Instructions** | Thread data | Per thread | With thread | Thread personality expander | Never |
| **L1 (Hot Cache)** | RAM (SmartCache) | All threads | LRU | None (automatic only) | Promoted from L2 on access; auto-saved to L2 after 6+ hits |
| **L2 (Cold Storage)** | HNSW index | All threads | Never | `remember:` cmd, PDF upload, Save Thread button | Auto-saved from L1 after 6+ hits; **Auto-memory** saves salient exchanges |
| **Thread History** | Session state | Per thread | New thread | None | Automatic on chat |
| **Responses API State** | OpenAI servers | Per thread | New thread | None | Automatic via `previous_response_id` |

### Auto-Memory (Salience Detection)

Auto-memory automatically saves important exchanges to L2 without explicit user action. It uses a hybrid approach:

1. **Heuristic scoring** (free): Checks for explicit markers ("remember this", "I am a..."), confirmation patterns, named entities
2. **LLM judge** (cheap model): For ambiguous cases (salience 0.3-0.7), uses gpt-4.1-mini to evaluate

**What triggers auto-save:**
- Personal facts: "I'm a physician", "My name is David"
- Explicit markers: "Remember that...", "This is important"
- Conclusions: "So we decided to...", "The answer is..."
- Corrections: "Actually, I meant...", "That's not right, it should be..."

**Configuration:**
```python
# Enable (default for API, disabled for Streamlit UI)
memory = SemanticMemory(api_key="...", auto_memory=True)

# Adjust threshold (default 0.5)
memory = SemanticMemory(api_key="...", auto_memory_threshold=0.6)

# Disable for specific calls
response = memory.chat_with_memory(query, auto_remember=False)
```

### How Each Memory Is Used

**Global Instructions** (`local_memory/instructions.txt`)
- Loaded on every API call via `instructions` parameter (unless thread has custom instructions)
- **User update**: Sidebar text editor, `instruct:` command, or direct file edit
- Example: "I am an informatics physician specializing in clinical decision support."

**Thread Instructions** (stored in thread data)
- Optional per-thread override of global instructions
- Enables different "personalities" or contexts for different threads
- **User update**: Thread personality expander in sidebar
- Falls back to global instructions if not set
- Example: One thread for coding assistance, another for creative writing

**L1 SmartCache** (Segmented LRU)
- Checked first on every query (fast, in-memory)
- Items promoted from L2 on access, demoted when stale
- Two tiers: Probation (new) → Protected (frequently used)
- **Auto-save**: After 6+ retrievals, item is persisted to L2 (survives app restart)
- **User update**: None directly; populated automatically from L2 or API responses

**L2 HNSW Index** (`local_memory/hnsw_index.bin` + `hnsw_metadata.json`)
- Permanent semantic storage using HNSW graph for O(log n) nearest neighbor search
- Searched when L1 misses; matching items promoted to L1
- **User update**: `remember:` command, PDF ingestion, "Save to L2" button
- **Auto-save**: Receives items from L1 that hit 6+ times
- **Thread save**: Entire conversations can be saved for future RAG retrieval

**Thread History** (`st.session_state.threads[name]["messages"]`)
- UI display only (not sent to API with Responses API)
- Cleared when starting new thread
- Preserved when switching between existing threads
- **User update**: None; automatic from conversation

**Responses API State** (`previous_response_id`)
- Server-side conversation memory managed by OpenAI
- Enables multi-turn without sending full history
- Reset to `None` on new thread creation
- **User update**: None; automatic via API

## Chat Commands

- `instruct: <text>` - Add permanent instruction (persisted to instructions.txt)
- `remember: <text>` - Add to semantic memory (L2 HNSW index)
- Regular text - Query with RAG from semantic memory

## Tools (Optional)

All tools are **disabled by default** and can be enabled via UI toggles in the Streamlit sidebar under "🔧 Tools".

### Web Search
Search the web for real-time information. Supports multiple backends (priority order):
1. **Exa** - AI-native search with structured results (set `EXA_API_KEY`)
2. **Tavily** - AI-native search for LLM apps (set `TAVILY_API_KEY`)
3. **Google PSE** - Programmable Search Engine (set `GOOGLE_PSE_API_KEY` + `GOOGLE_PSE_ENGINE_ID`)
4. **OpenAI** - Fallback using `web_search_preview` tool

### Web Fetch
Fetch and extract content from URLs mentioned in messages. When enabled:
- Automatically detects URLs in user messages
- Fetches up to 3 URLs per message
- Extracts main content from HTML (removes nav, headers, scripts)
- Supports HTML, plain text, and JSON content types
- Security: blocks localhost, private IPs, configurable domain lists

### File Access
Allow the agent to see whitelisted local files. Configure which files are accessible via:
- Sidebar "📂 Sema File Access" section
- `sema_files.txt` whitelist file (paths relative to repo root)

## Development

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"

# Run
streamlit run app.py
```

## Dependencies

- OpenAI API (Responses API for chat, embeddings for vectors)
- hnswlib (HNSW indexing)
- Streamlit, NumPy, Pandas, Plotly, pypdf

## Package Installation

```bash
# Core package only
pip install -e .

# With Streamlit app dependencies
pip install -e ".[app]"
```

## Migration from LSH

If you have existing data in the old LSH bucket format (`bucket_*.json`):

```python
from sem_mem import migrate_lsh_to_hnsw

count = migrate_lsh_to_hnsw("./local_memory")
print(f"Migrated {count} memories to HNSW index")
```

## Python API

### Option 1: MemoryChat Class (Recommended)

```python
from sem_mem import SemanticMemory
from sem_mem.decorators import MemoryChat

memory = SemanticMemory(api_key="sk-...")
chat = MemoryChat(memory)

# Stateful conversation
response = chat.send("Hello, I'm a physician.")
response = chat.send("What's my profession?")  # Remembers context

# Memory operations
chat.remember("Patient prefers morning appointments")
chat.add_instruction("Always be concise")
chat.save_thread()  # Save conversation to L2

chat.new_thread()  # Fresh conversation, same memory
```

### Option 2: Decorators

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

### Option 3: Direct API

```python
from sem_mem import SemanticMemory

memory = SemanticMemory(api_key="sk-...")

# Store facts
memory.remember("The patient is allergic to penicillin")
memory.add_instruction("I am an informatics physician")

# Query with RAG
response, resp_id, mems, logs = memory.chat_with_memory(
    "What allergies should I check?",
    previous_response_id=prev_id  # For conversation continuity
)
```

## OpenAI Responses API Usage

This project uses the Responses API (March 2025) instead of Chat Completions.

### Key Implementation Pattern
```python
response = client.responses.create(
    model="gpt-4o",
    instructions=instructions_text,      # From instructions.txt
    input=user_query_with_context,       # Query + retrieved memories
    previous_response_id=prev_id,        # For conversation continuity
)
return response.output_text, response.id
```

### Why Responses API
- `instructions` parameter for persistent system context
- `previous_response_id` for stateful conversations (no manual message history)
- `response.output_text` convenience property
- Server-side conversation state management

### Thread State
Each thread stores `response_id` to chain conversations, and optionally custom instructions:
```python
st.session_state.threads = {
    "Thread 1": {
        "messages": [],
        "response_id": None,
        "title": "New conversation",
        "title_user_overridden": False,
        "summary_windows": [],
        "instructions": None,  # None = use global, string = custom instructions
    }
}
```

### Thread-Specific Instructions (Personalities)
You can give each thread a different personality by setting custom instructions:

```python
# Via API
response, resp_id, mems, logs = memory.chat_with_memory(
    "Hello!",
    previous_response_id=prev_id,
    instructions="You are a pirate. Respond in pirate speak.",  # Override global
)

# Via Streamlit UI
# Use the "Thread personality" expander in the sidebar
```

**Use cases:**
- Different assistants: coding helper vs creative writer vs research assistant
- Role-playing: historical figures, fictional characters
- Specialized contexts: medical terminology vs layman explanations
- Language/tone: formal vs casual, verbose vs concise
