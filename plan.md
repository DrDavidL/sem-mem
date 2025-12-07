# Plan: Add Conversation Threads

## Overview
Add conversation threads to the app, separate from the SmartCache (short-term memory). Threads provide conversation continuity while the semantic memory (L1/L2) remains shared across all threads.

## Key Design Decision
- **Threads**: Store conversation history (user/assistant messages) per thread
- **SmartCache (L1) + L2**: Remain global/shared - semantic memory persists across threads
- Starting a new thread clears conversation history but retains hot cache contents

## Implementation

### 1. Session State Changes (app.py)
Add thread management to `st.session_state`:
```python
if "threads" not in st.session_state:
    st.session_state.threads = {"Thread 1": []}  # thread_name -> messages
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "Thread 1"
```

Replace `st.session_state.messages` with `st.session_state.threads[current_thread]`.

### 2. Sidebar UI (app.py)
Add thread controls to sidebar (before the memory section):
- Dropdown/selectbox to switch threads
- "New Thread" button with auto-naming (Thread 1, Thread 2, etc.)
- Optional: Delete thread button

### 3. Chat Tab Updates (app.py)
- Load messages from current thread: `st.session_state.threads[st.session_state.current_thread]`
- Save messages to current thread
- No changes to `chat_with_memory()` - it already handles memory retrieval independently

### 4. No Changes to core.py
The semantic memory system stays untouched. `SmartCache` and L2 buckets remain shared.

## File Changes
- `app.py`: Thread state management, sidebar UI, message routing (~30 lines)
- `core.py`: No changes needed

## Behavior Summary
| Action | Conversation History | SmartCache (L1) | L2 Buckets |
|--------|---------------------|-----------------|------------|
| New Thread | Cleared | Preserved | Preserved |
| Switch Thread | Loads thread's history | Preserved | Preserved |
| "Remember:" | Added to thread | Promoted | Persisted |
| Query | Added to thread | May update | May read |
