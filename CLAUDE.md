# Sem-Mem Project

Tiered semantic memory system for AI agents using HNSW-based vector indexing.

## Recent Improvements

These changes make Sem-Mem more inspectable, trustworthy, and production-friendly: you can see how memory evolved, why certain facts were preferred, and what each background pass actually did.

- **Structured progress logging** - Each consolidation run records what changed and why in a machine-readable log (`progress_log.jsonl`), making long-running memory evolution auditable and debuggable.

- **Project manifests & progress summaries** - Per-project / per-thread "manifest" and "what we did / what's next" memories (`save_project_manifest()`, `append_thread_progress()`) make it easy for both humans and agents to re-enter complex work without re-reading long threads.

- **Smarter status & progress queries** - Queries like "what's the status of X?" preferentially surface manifests and progress logs, so you get a concise state-of-the-world answer instead of raw history dumps.

- **Explicit conflict handling & precedence** - Conflicting memories (fact vs. correction, old vs. new policy) are handled via clear rules: explicit correction entries, recency and utility bias, and no silent overwrites. See [Conflict Handling & Precedence](#conflict-handling--precedence).

- **Contradiction surfacing, not silent fixes** - Detected contradictions are logged (with memory IDs and summaries) to `contradictions.json` for human review, providing a clear trail of "what disagrees with what" instead of hidden heuristics.

- **Stronger guarantees about consolidation behavior** - Consolidation is explicitly offline, human-scheduled, and non-agentic, with hard caps on changes per run (`CONSOLIDATION_MAX_NEW_PATTERNS`) and clear configuration describing its role.

## Architecture

- **SmartCache (L1)**: Segmented LRU in RAM with Protected/Probation tiers
- **L2 Storage**: HNSW index (`hnsw_index.bin` + `hnsw_metadata.json`) for O(log n) semantic search
- **Embeddings**: Pluggable embedding providers (OpenAI, local via sentence-transformers, Ollama)
- **Instructions**: Persistent system instructions in `local_memory/instructions.txt`
- **UI**: Streamlit app (`app.py`)

## Embedding Providers

Sem-Mem supports multiple embedding providers. The recommended setup is local embeddings via sentence-transformers (no API costs, no rate limits).

### Available Providers

| Provider | Model | Dimension | API Key Required |
|----------|-------|-----------|------------------|
| `sentence-transformers` (recommended) | Qwen/Qwen3-Embedding-0.6B | 1024 | No |
| `local` (alias) | Qwen/Qwen3-Embedding-0.6B | 1024 | No |
| `openai` | text-embedding-3-small | 1536 | Yes |
| `ollama` | nomic-embed-text | 768 | No |
| `google` | text-embedding-004 | 768 | Yes |

### Switching to Local Embeddings

To switch from OpenAI to local embeddings (recommended to avoid rate limits):

```bash
# 1. Install sentence-transformers
pip install sentence-transformers torch

# 2. Run migration script (backs up existing index, re-embeds all memories)
python scripts/migrate_embeddings.py --provider sentence-transformers

# 3. Update your .env file
SEMMEM_EMBEDDING_PROVIDER=sentence-transformers

# 4. Restart your application
```

### Configuration

Set embedding provider via environment variables:

```bash
# In .env file or environment
SEMMEM_EMBEDDING_PROVIDER=sentence-transformers  # or: openai, local, ollama, google
SEMMEM_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B  # optional, uses provider default

# Chat still uses OpenAI (or other provider)
SEMMEM_CHAT_PROVIDER=openai
```

Or programmatically:

```python
from sem_mem import SemanticMemory

# Use local embeddings with OpenAI chat
memory = SemanticMemory(
    api_key="sk-...",  # For chat
    embedding_provider="sentence-transformers",
)
```

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

### Web Fetch (Playwright-based)
Fetch and extract content from URLs using Playwright for JavaScript rendering.

**Playwright Benefits:**
- **JavaScript rendering** - Handles SPAs and dynamic content
- **Parallel fetching** - Fetches multiple URLs concurrently (up to 5 pages)
- **Better content extraction** - Waits for content to load before extracting
- **Automatic fallback** - Falls back to requests-based fetching if Playwright unavailable

**Installation:**
```bash
pip install playwright
playwright install chromium
```

**Passive Mode** (context injection):
- Automatically detects URLs in user messages
- Fetches up to 5 URLs in parallel via Playwright
- Content injected into context before LLM call

**Active/Agentic Mode** (tool calling):
- LLM can proactively request URL fetches when needed
- Useful for real-time data: stock prices, weather, news, documentation
- Uses OpenAI Responses API tool calling with automatic result feeding
- Up to 3 tool call iterations to prevent infinite loops

**Search + Fetch Workflow:**
The recommended pattern is to combine web search with parallel content fetching:
1. Search via Exa/Tavily/Google PSE to get URLs
2. Fetch all URLs in parallel via Playwright
3. Inject full page content as context for the LLM

```python
from sem_mem import SemanticMemory

memory = SemanticMemory(api_key="...", web_search=True, web_fetch=True)

# Method 1: Combined search and fetch
context, logs = memory.search_and_fetch("Python asyncio best practices", num_results=5)
# Returns formatted context with search results + full page content

# Method 2: Direct URL fetching (parallel)
results = memory.fetch_urls(["https://example1.com", "https://example2.com"])
for content, success in results:
    if success:
        print(content)
```

**Features:**
- Extracts main content from HTML (removes nav, headers, scripts)
- Supports HTML, plain text, and JSON content types
- Security: blocks localhost, private IPs, configurable domain lists
- Configurable concurrency (default: 5 parallel pages)
- Configurable timeout (default: 30s per page)

**Configuration:**
```
WEB_FETCH_USE_PLAYWRIGHT=true   # Use Playwright (default if installed)
WEB_FETCH_USE_PLAYWRIGHT=false  # Force requests-based fetching
```

**Example flow (agentic):**
1. User asks: "What's NVDA's current stock price?"
2. LLM requests `web_fetch(url="https://finance.yahoo.com/quote/NVDA/")`
3. System fetches URL via Playwright, extracts content
4. LLM receives content and provides answer with actual price

### File Access (Agentic)
Read whitelisted local files. Works in two modes:

**Passive Mode** (context injection):
- Tells the AI which files are available in the whitelist
- File list included in system context

**Active/Agentic Mode** (tool calling):
- LLM can proactively request file content when needed
- Only files in the whitelist can be read
- Uses OpenAI Responses API tool calling with automatic result feeding
- Supports text files and PDFs (with text extraction)

**Configuration:**
- Sidebar "📂 Sema File Access" section
- `sema_files.txt` whitelist file (paths relative to repo root)
- Can whitelist individual files or entire directories

**Example flow (agentic):**
1. User asks: "What does the core.py file do?"
2. LLM requests `file_read(path="sem_mem/core.py")`
3. System reads file content (if whitelisted)
4. LLM receives content and provides explanation

**Security:**
- Only whitelisted files can be read
- Path traversal attacks are blocked
- Binary files (except PDF) are rejected

### Stock Price (Agentic)
Get real-time stock prices and market data using Polygon.io API.

**Automatic enablement:**
- Automatically enabled when `POLYGON_API_KEY` is set in `.env`
- No UI toggle needed - the tool appears when API key is configured

**Configuration:**
Add to your `.env` file:
```
POLYGON_API_KEY=your_polygon_api_key_here
```

**Data returned:**
- Current/latest trade price
- Daily change (amount and percentage)
- Open, high, low prices
- Previous close
- Trading volume
- Timestamp

**Example flow:**
1. User asks: "What's Apple's stock price?"
2. LLM requests `stock_price(ticker="AAPL")`
3. System fetches data from Polygon.io API
4. LLM receives: price, change, OHLC, volume
5. LLM provides formatted response

**Supported tickers:**
- All major US stock exchanges (NYSE, NASDAQ, etc.)
- Case-insensitive ticker lookup
- Examples: AAPL, MSFT, GOOGL, NVDA, TSLA, AMZN

**API limits:**
- Free tier: 5 requests/minute, end-of-day data
- Paid tiers: Real-time data, higher rate limits
- See: https://polygon.io/pricing

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
chat.remember("Spouse prefers early AM flights.")
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

## Conflict Handling & Precedence

When memories contain conflicting information, the system follows these rules:

### How Conflicts Are Handled

1. **Corrections are stored as separate memories** - When a user corrects earlier information (e.g., "Actually, my name is spelled differently"), a new memory is created with `kind="correction"`. The old memory is not deleted.

2. **Corrections are surfaced after older facts** - In `_format_memories_for_display()`, regular memories appear first, then corrections appear last with `[CORRECTION]` prefix. This ensures the model sees the correction as the authoritative version.

3. **Time-decay + utility nudge toward recent corrections** - Recent memories score higher via time-decay. Corrections that get positive feedback accumulate higher utility scores, making them more likely to be retrieved.

4. **No silent deletion** - Old memories are never deleted when contradicted. Instead:
   - Consolidation may demote them via `record_outcome(id, "failure")`
   - Time decay naturally reduces their retrieval priority
   - They remain available for audit/history

### Contradiction Detection

The consolidator flags potential contradictions for human review:
- Stored in `local_memory/contradictions.json`
- Each entry: `{"ids": [id1, id2], "summary": "...", "status": "pending_review", "detected_at": "..."}`
- Contradictions are **never auto-resolved** - humans must review and decide

### Memory Kinds for Retrieval

| Kind | Description | Retrieval Behavior |
|------|-------------|-------------------|
| `fact` | General information | Standard scoring |
| `identity` | Who the user is | Standard scoring |
| `preference` | User preferences | Standard scoring |
| `decision` | Decisions made together | Standard scoring |
| `correction` | Updates to prior info | Displayed last with `[CORRECTION]` prefix |
| `pattern` | Stable principle from consolidation | Slightly boosted via utility |
| `project_manifest` | Project state/goals | Boosted for "status" queries |
| `progress_log` | Session progress notes | Boosted for "status" queries |

### Example: Correction Flow

```python
# User initially says:
"My favorite color is blue"
# -> Stored as kind="fact"

# Later user corrects:
"Actually, my favorite color is green"
# -> Stored as kind="correction", old memory remains

# On retrieval for "What's my favorite color?":
# -> Returns: ["My favorite color is blue", "[CORRECTION] My favorite color is green"]
# -> Model sees correction last, treats it as authoritative
```
