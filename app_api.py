"""
Streamlit app that uses the FastAPI server as backend.

This demonstrates the API and allows the UI to run separately from the memory system.

Usage:
    1. Start the API server: uvicorn server:app --reload
    2. Run this app: streamlit run app_api.py
"""

import streamlit as st
import httpx
import os
import copy
from datetime import datetime, timezone

from sem_mem.config import (
    AUTO_THREAD_RENAME_ENABLED,
    AUTO_THREAD_RENAME_MIN_USER_MESSAGES,
    ON_DELETE_THREAD_BEHAVIOR,
)
from sem_mem.file_access import (
    load_whitelist_raw,
    load_whitelist,
    get_file_info,
    add_to_whitelist,
    remove_from_whitelist,
    get_suggested_paths,
)

# Page Config
st.set_page_config(page_title="Sem-Mem (API Client)", layout="wide")

# --- API Configuration ---
API_URL = os.getenv("SEMMEM_API_URL", "http://localhost:8000")


def api_request(method: str, endpoint: str, timeout: float = 120.0, **kwargs) -> dict:
    """Make a request to the API server."""
    url = f"{API_URL}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        st.error(f"Cannot connect to API server at {API_URL}. Is it running?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return {}


def api_request_raw(method: str, endpoint: str, timeout: float = 120.0, **kwargs):
    """Make a request and return raw response (for file downloads)."""
    url = f"{API_URL}{endpoint}"
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
    except httpx.ConnectError:
        st.error(f"Cannot connect to API server at {API_URL}. Is it running?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return None


# --- Check API Health ---
@st.cache_data(ttl=5)
def check_health():
    """Check if API is healthy."""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{API_URL}/health")
            return response.status_code == 200
    except Exception:
        return False


if not check_health():
    st.error(f"API server not available at {API_URL}")
    st.info("Start the server with: `uvicorn server:app --reload`")
    st.stop()


# --- Thread Management ---
# Thread structure matches app.py:
#   {
#       "messages": [],              # List of {role, content} dicts
#       "response_id": None,         # OpenAI Responses API ID for continuity
#       "title": "New conversation", # Display title (auto-generated or user-set)
#       "title_user_overridden": False,  # If True, auto-rename won't touch it
#       "instructions": None,        # Thread-specific instructions (None = use global)
#   }

def _create_empty_thread() -> dict:
    """Create a new empty thread with default structure."""
    return {
        "messages": [],
        "response_id": None,
        "title": "New conversation",
        "title_user_overridden": False,
        "instructions": None,  # None = use global instructions
    }


# --- Thread Persistence ---
def _load_persisted_threads() -> dict:
    """Load threads from API server."""
    try:
        result = api_request("GET", "/threads", timeout=5.0)
        if result and result.get("threads"):
            return result["threads"]
    except Exception:
        pass  # Fallback to empty
    return {}


def _persist_threads_if_changed():
    """Save threads to API server if they changed (debounced)."""
    if "threads" not in st.session_state:
        return

    current = st.session_state.threads
    last = st.session_state.get("_last_saved_threads", {})

    if current != last:
        try:
            api_request("POST", "/threads", json={"threads": current}, timeout=10.0)
            st.session_state._last_saved_threads = copy.deepcopy(current)
        except Exception:
            pass  # Don't break UI on save failures


# Load threads on startup (from API or default)
if "threads" not in st.session_state:
    persisted = _load_persisted_threads()
    if persisted:
        st.session_state.threads = persisted
        st.session_state._last_saved_threads = copy.deepcopy(persisted)
    else:
        st.session_state.threads = {"Thread 1": _create_empty_thread()}
        st.session_state._last_saved_threads = {}

if "current_thread" not in st.session_state:
    # Pick first available thread or default
    if st.session_state.threads:
        st.session_state.current_thread = list(st.session_state.threads.keys())[0]
    else:
        st.session_state.current_thread = "Thread 1"


def _should_auto_rename(thread: dict) -> bool:
    """Check if thread is eligible for auto-rename."""
    if not AUTO_THREAD_RENAME_ENABLED:
        return False
    if thread.get("title_user_overridden", False):
        return False
    current_title = thread.get("title", "New conversation")
    if current_title not in ("New conversation", "", None):
        return False
    user_msg_count = sum(1 for m in thread["messages"] if m["role"] == "user")
    return user_msg_count >= AUTO_THREAD_RENAME_MIN_USER_MESSAGES


def _get_thread_display_name(thread_key: str, thread: dict) -> str:
    """Get display name for thread (title if set, else key)."""
    title = thread.get("title", "")
    if title and title != "New conversation":
        return title
    return thread_key


def _generate_thread_title_via_api(messages: list) -> str:
    """Generate a thread title using the chat API."""
    if not messages:
        return ""

    # Build a summarization request
    conversation_text = "\n".join([
        f"{m['role'].upper()}: {m['content'][:200]}"
        for m in messages[:10]  # Limit to first 10 messages
    ])

    # Use the chat endpoint with a title generation prompt
    result = api_request(
        "POST",
        "/chat",
        json={
            "query": f"Generate a short title (max 8 words) for this conversation. Reply with ONLY the title, nothing else:\n\n{conversation_text}",
        },
        timeout=30.0
    )

    if result and result.get("response"):
        title = result["response"].strip().strip('"\'')
        # Truncate if too long
        words = title.split()
        if len(words) > 8:
            title = " ".join(words[:8])
        return title
    return ""


def _handle_delete_thread(thread_name: str, save_summary: bool) -> bool:
    """Delete a thread, optionally saving a farewell summary first."""
    if thread_name not in st.session_state.threads:
        return False

    thread_data = st.session_state.threads[thread_name]

    # Generate and store farewell summary if requested
    if save_summary:
        messages = thread_data.get("messages", [])
        if len(messages) >= 2:
            # Build conversation text for summarization
            conversation_text = "\n".join([
                f"{m['role'].upper()}: {m['content'][:500]}"
                for m in messages
            ])

            # Use chat API to generate summary
            result = api_request(
                "POST",
                "/chat",
                json={
                    "query": f"""Summarize this conversation for long-term memory. Focus on:
- Key decisions or conclusions
- User preferences
- Important context

Conversation:
{conversation_text[:4000]}

Write a concise summary.""",
                },
                timeout=60.0
            )

            if result and result.get("response"):
                summary = result["response"]
                # Store the summary
                api_request(
                    "POST",
                    "/remember",
                    json={
                        "text": summary,
                        "metadata": {
                            "type": "farewell_summary",
                            "thread_name": thread_name,
                            "thread_title": thread_data.get("title", thread_name),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )

    # Delete the thread
    del st.session_state.threads[thread_name]

    # Switch to another thread or create a new one
    remaining_threads = list(st.session_state.threads.keys())
    if remaining_threads:
        st.session_state.current_thread = remaining_threads[0]
    else:
        st.session_state.threads["Thread 1"] = _create_empty_thread()
        st.session_state.current_thread = "Thread 1"

    # Persist the change
    _persist_threads_if_changed()

    return True


# --- Sidebar ---
with st.sidebar:
    st.caption(f"Connected to: `{API_URL}`")

    # Thread Controls
    st.header("Threads")
    thread_names = list(st.session_state.threads.keys())

    # Build display names for selectbox
    thread_display_names = [
        _get_thread_display_name(name, st.session_state.threads[name])
        for name in thread_names
    ]

    # Map display name back to thread key
    current_display = _get_thread_display_name(
        st.session_state.current_thread,
        st.session_state.threads[st.session_state.current_thread]
    )

    selected_display = st.selectbox(
        "Current Thread",
        thread_display_names,
        index=thread_display_names.index(current_display) if current_display in thread_display_names else 0
    )

    # Find the actual thread key from display name
    selected_idx = thread_display_names.index(selected_display)
    selected_thread = thread_names[selected_idx]

    if selected_thread != st.session_state.current_thread:
        st.session_state.current_thread = selected_thread
        st.rerun()

    # Thread rename UI
    current_thread_data = st.session_state.threads[st.session_state.current_thread]
    with st.expander("Rename thread", expanded=False):
        current_title = current_thread_data.get("title", "New conversation")
        new_title = st.text_input(
            "Thread title",
            value=current_title,
            key="thread_title_input",
            label_visibility="collapsed",
            placeholder="Enter a title..."
        )
        if st.button("Save title"):
            if new_title and new_title != current_title:
                current_thread_data["title"] = new_title.strip()
                current_thread_data["title_user_overridden"] = True
                _persist_threads_if_changed()
                st.success("Title saved!")
                st.rerun()

    # Thread delete UI
    with st.expander("🗑️ Delete thread", expanded=False):
        has_messages = len(current_thread_data.get("messages", [])) >= 2

        if ON_DELETE_THREAD_BEHAVIOR == "prompt" and has_messages:
            st.caption("This thread has content. Save a summary before deleting?")
            del_col1, del_col2 = st.columns(2)
            with del_col1:
                if st.button("Delete & Save", type="primary", use_container_width=True):
                    with st.spinner("Summarizing..."):
                        _handle_delete_thread(st.session_state.current_thread, save_summary=True)
                    st.toast("Thread deleted. Summary saved to memory.", icon="✅")
                    st.rerun()
            with del_col2:
                if st.button("Just Delete", use_container_width=True):
                    _handle_delete_thread(st.session_state.current_thread, save_summary=False)
                    st.toast("Thread deleted.", icon="🗑️")
                    st.rerun()
        elif ON_DELETE_THREAD_BEHAVIOR == "always_save" and has_messages:
            st.caption("A summary will be saved to memory before deletion.")
            if st.button("Delete Thread", type="primary", use_container_width=True):
                with st.spinner("Summarizing and deleting..."):
                    _handle_delete_thread(st.session_state.current_thread, save_summary=True)
                st.toast("Thread deleted. Summary saved to memory.", icon="✅")
                st.rerun()
        else:
            if st.button("Delete Thread", type="secondary", use_container_width=True):
                _handle_delete_thread(st.session_state.current_thread, save_summary=False)
                st.toast("Thread deleted.", icon="🗑️")
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ New Thread"):
            new_id = len(st.session_state.threads) + 1
            new_name = f"Thread {new_id}"
            st.session_state.threads[new_name] = _create_empty_thread()
            st.session_state.current_thread = new_name
            _persist_threads_if_changed()
            st.rerun()
    with col2:
        current_messages = st.session_state.threads[st.session_state.current_thread]["messages"]
        if st.button("Save to L2", disabled=len(current_messages) == 0):
            with st.spinner("Saving thread..."):
                result = api_request(
                    "POST",
                    "/threads/save",
                    params={"thread_name": st.session_state.current_thread},
                    json=current_messages
                )
            if result:
                st.success(f"Saved {result.get('chunks_saved', 0)} chunks!")

    st.divider()

    # Model Selection (fetched from API)
    st.header("🤖 Model")
    model_config = api_request("GET", "/model")
    if model_config:
        available_models = model_config.get("available_models", ["gpt-5.1", "gpt-4.1"])
        current_model = model_config.get("current_model", "gpt-5.1")
        reasoning_efforts = model_config.get("reasoning_efforts", ["low", "medium", "high"])
        current_reasoning = model_config.get("current_reasoning_effort", "low")

        selected_model = st.selectbox(
            "Chat Model",
            available_models,
            index=available_models.index(current_model) if current_model in available_models else 0,
            help="gpt-5.1 is a reasoning model (like o3)"
        )

        # Update model if changed
        if selected_model != current_model:
            api_request("PUT", f"/model?model={selected_model}")
            st.rerun()

        # Show reasoning effort only for reasoning models
        if selected_model in ("gpt-5.1", "o1", "o3"):
            selected_effort = st.select_slider(
                "Reasoning Effort",
                options=reasoning_efforts,
                value=current_reasoning if current_reasoning in reasoning_efforts else "low",
                help="Higher = more thorough but slower"
            )
            if selected_effort != current_reasoning:
                api_request("PUT", f"/model?model={selected_model}&reasoning_effort={selected_effort}")
                st.rerun()

    # Web Search Toggle
    if "web_search" not in st.session_state:
        st.session_state.web_search = False

    web_search_enabled = st.toggle(
        "🌐 Web Search",
        value=st.session_state.web_search,
        help="Enable web search for real-time information"
    )
    if web_search_enabled != st.session_state.web_search:
        st.session_state.web_search = web_search_enabled

    st.divider()

    # Instructions Editor
    st.header("📋 Instructions")
    st.caption("Permanent context for every request:")

    # Fetch current instructions
    instructions_data = api_request("GET", "/instructions")
    current_instructions = instructions_data.get("instructions", "")

    new_instructions = st.text_area(
        "Edit instructions",
        value=current_instructions,
        height=100,
        label_visibility="collapsed"
    )
    if st.button("Save Instructions"):
        api_request("PUT", "/instructions", json={"instruction": new_instructions})
        st.success("Instructions saved!")
        st.cache_data.clear()

    st.divider()
    st.header("🧠 Distributed Memory")

    # PDF Upload
    with st.expander("📚 Digest Knowledge (PDF)", expanded=True):
        uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_pdf and st.button("Ingest PDF"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                result = api_request("POST", "/upload/pdf", files=files)
            if result:
                st.success(f"Processed {result.get('pages', 0)} pages, stored {result.get('chunks_stored', 0)} chunks!")
                st.cache_data.clear()

    # Batch Remember
    with st.expander("📥 Batch Remember"):
        batch_text = st.text_area("Enter facts (one per line):", height=100)
        if st.button("Remember All") and batch_text.strip():
            lines = [l.strip() for l in batch_text.split("\n") if l.strip()]
            with st.spinner(f"Storing {len(lines)} memories..."):
                result = api_request("POST", "/remember/batch", json={"texts": lines})
            if result:
                st.success(f"Added {result.get('added', 0)} new memories!")
                st.cache_data.clear()

    st.divider()

    # Cache State
    st.subheader("🔥 Hot Cache (L1)")
    cache_data = api_request("GET", "/cache")
    if cache_data:
        protected = cache_data.get("protected", [])
        probation = cache_data.get("probation", [])

        if not protected and not probation:
            st.info("Cache is empty")
        else:
            if protected:
                st.caption("Protected:")
                for item in protected[:3]:
                    st.text(f"  {item[:35]}...")
            if probation:
                st.caption("Probation:")
                for item in probation[:3]:
                    st.text(f"  {item[:35]}...")

    st.divider()

    # Stats
    stats = api_request("GET", "/stats")
    if stats:
        st.metric("L2 Memories", stats.get("total_l2_memories", 0))
        st.metric("L1 Cache Size", stats.get("l1_cache_size", 0))

    st.divider()

    # Backup/Restore
    st.header("💾 Backup & Restore")

    with st.expander("Create Backup", expanded=False):
        backup_name = st.text_input(
            "Backup name (optional)",
            placeholder="my-backup (uses timestamp if empty)",
            key="backup_name_input"
        )
        if st.button("Create Backup", key="create_backup_btn"):
            with st.spinner("Creating backup..."):
                params = {}
                if backup_name.strip():
                    params["backup_name"] = backup_name.strip()
                result = api_request("POST", "/backup/create", params=params)
            if result:
                st.success(f"Backup created: {result.get('path', '')}")
                st.caption(f"Memories: {result.get('stats', {}).get('memory_count', 0)}, Threads: {result.get('stats', {}).get('thread_count', 0)}")

    with st.expander("Restore from Backup", expanded=False):
        # Fetch available backups
        backups = api_request("GET", "/backup/list")
        if backups:
            backup_options = [b["name"] for b in backups]
            selected_backup = st.selectbox(
                "Select backup",
                options=backup_options,
                key="restore_backup_select"
            )

            # Show backup details
            if selected_backup:
                backup_info = next((b for b in backups if b["name"] == selected_backup), None)
                if backup_info:
                    st.caption(f"Created: {backup_info.get('created_at', 'unknown')}")
                    st.caption(f"Memories: {backup_info.get('memory_count', 0)}, Threads: {backup_info.get('thread_count', 0)}")

            merge_mode = st.checkbox("Merge with existing data", value=False, key="restore_merge")
            re_embed = st.checkbox("Re-embed vectors (if model changed)", value=False, key="restore_reembed")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Restore", type="primary", key="restore_btn"):
                    with st.spinner("Restoring..."):
                        result = api_request(
                            "POST",
                            "/backup/restore",
                            json={
                                "backup_name": selected_backup,
                                "merge": merge_mode,
                                "re_embed": re_embed
                            }
                        )
                    if result:
                        st.success("Restore complete!")
                        st.caption(f"Added: {result.get('memories_added', 0)}, Skipped: {result.get('memories_skipped', 0)}")
                        st.caption(f"Threads: {result.get('threads_restored', 0)}, Instructions: {result.get('instructions_action', '')}")
                        # Reload threads from server
                        st.session_state.pop("threads", None)
                        st.session_state.pop("_last_saved_threads", None)
                        st.cache_data.clear()
                        st.rerun()
            with col2:
                if st.button("Delete Backup", type="secondary", key="delete_backup_btn"):
                    result = api_request("DELETE", f"/backup/{selected_backup}")
                    if result and result.get("deleted"):
                        st.success("Backup deleted!")
                        st.rerun()
        else:
            st.info("No backups available")

    with st.expander("Export Data", expanded=False):
        st.caption("Download all memory data as JSON")
        include_vectors = st.checkbox("Include vectors", value=True, key="export_vectors")
        include_threads = st.checkbox("Include threads", value=True, key="export_threads")

        if st.button("Export", key="export_btn"):
            with st.spinner("Exporting..."):
                result = api_request(
                    "GET",
                    "/backup/export",
                    params={
                        "include_vectors": include_vectors,
                        "include_threads": include_threads
                    }
                )
            if result:
                import json
                export_json = json.dumps(result, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=export_json,
                    file_name=f"sem-mem-export-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

    st.divider()

    # Sema File Access Management
    st.header("📂 Sema File Access")
    st.caption("Manage which files Sema (Semantic Memory Agent) can see")

    with st.expander("Current whitelist", expanded=False):
        raw_entries = load_whitelist_raw()
        if raw_entries:
            for entry in raw_entries:
                st.text(f"• {entry}")

            # Show expanded file count
            allowed_files = load_whitelist()
            st.caption(f"({len(allowed_files)} files total)")
        else:
            st.info("No files whitelisted yet")

        if st.button("🔄 Reload", key="reload_whitelist"):
            st.rerun()

    with st.expander("Add to whitelist", expanded=False):
        new_path = st.text_input(
            "Path (relative to repo root)",
            placeholder="e.g., sem_mem/core.py or docs/",
            key="sema_add_path"
        )

        if st.button("Add", key="sema_add_btn"):
            if new_path:
                if add_to_whitelist(new_path):
                    st.success(f"Added: {new_path}")
                    st.rerun()
                else:
                    st.warning("Already in whitelist")

        suggestions = get_suggested_paths()
        if suggestions:
            st.caption("Suggestions:")
            selected_suggestion = st.selectbox(
                "Quick add",
                options=[""] + suggestions,
                key="sema_suggestion",
                label_visibility="collapsed"
            )
            if selected_suggestion and st.button("Add selected", key="sema_add_suggestion"):
                if add_to_whitelist(selected_suggestion):
                    st.success(f"Added: {selected_suggestion}")
                    st.rerun()

    with st.expander("Remove from whitelist", expanded=False):
        raw_entries = load_whitelist_raw()
        if raw_entries:
            entry_to_remove = st.selectbox(
                "Select entry to remove",
                options=raw_entries,
                key="sema_remove_select"
            )
            if st.button("Remove", key="sema_remove_btn", type="secondary"):
                if remove_from_whitelist(entry_to_remove):
                    st.success(f"Removed: {entry_to_remove}")
                    st.rerun()
        else:
            st.info("No entries to remove")

# --- Main Interface ---
st.title("Sem-Mem: API Client")

tab1, tab2, tab3 = st.tabs(["💬 Chat", "📂 Files", "🔧 API Explorer"])

# TAB 1: Chat
with tab1:
    thread_data = st.session_state.threads[st.session_state.current_thread]
    messages = thread_data["messages"]

    # Chat container for message history (fixed height for stable layout)
    chat_container = st.container(height=500)

    with chat_container:
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input at fixed position below container
    if prompt := st.chat_input("Ask a question, 'remember: ...', or 'instruct: ...'"):
        messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 1. Instruct Command
        if prompt.lower().startswith("instruct:"):
            instruction = prompt.split(":", 1)[1].strip()
            api_request("POST", "/instructions", json={"instruction": instruction})
            response = f"✅ **Instruction added.**\n*\"{instruction}\"*"
            messages.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            st.cache_data.clear()
            _persist_threads_if_changed()

        # 2. Remember Command
        elif prompt.lower().startswith("remember:"):
            fact = prompt.split(":", 1)[1].strip()
            result = api_request("POST", "/remember", json={"text": fact})
            response = f"✅ **Stored.** \n*({result.get('message', '')})*"
            messages.append({"role": "assistant", "content": response})
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
            st.cache_data.clear()
            _persist_threads_if_changed()

        # 3. Chat with RAG
        else:
            prev_id = thread_data["response_id"]
            with st.spinner("Thinking..."):
                chat_payload = {
                    "query": prompt,
                    "previous_response_id": prev_id,
                    "web_search": st.session_state.web_search
                }
                result = api_request(
                    "POST",
                    "/chat",
                    json=chat_payload
                )

            if result:
                thread_data["response_id"] = result.get("response_id")
                response_text = result.get("response", "")
                logs = result.get("logs", [])
                retrieved_mems = result.get("memories", [])

                full_response = response_text

                if logs:
                    log_str = "\n".join([f"`{l}`" for l in logs])
                    full_response += f"\n\n**System Logs:**\n{log_str}"

                if retrieved_mems:
                    sources = "\n\n**Retrieved Context:**\n" + "\n".join([f"> *{m[:100]}...*" for m in retrieved_mems])
                    full_response += sources

                messages.append({"role": "assistant", "content": full_response})
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(full_response)

                # Persist thread after assistant response
                _persist_threads_if_changed()

        # Auto-rename thread if eligible
        if _should_auto_rename(thread_data):
            with st.spinner("Generating title..."):
                new_title = _generate_thread_title_via_api(messages)
            if new_title:
                thread_data["title"] = new_title.strip()
                _persist_threads_if_changed()
                st.rerun()

# TAB 2: Files (Sema's view)
with tab2:
    st.header("📂 Whitelisted Files")
    st.caption("Files Sema (Semantic Memory Agent) can access via the API")

    # Fetch files from API
    files_result = api_request("GET", "/files")

    if files_result:
        # Group by directory
        file_tree = {}
        for f in files_result:
            path = f["path"]
            parts = path.split("/")
            if len(parts) > 1:
                dir_name = parts[0]
            else:
                dir_name = "(root)"

            if dir_name not in file_tree:
                file_tree[dir_name] = []
            file_tree[dir_name].append(f)

        # Display as expandable sections
        for dir_name in sorted(file_tree.keys()):
            dir_files = file_tree[dir_name]
            with st.expander(f"📁 {dir_name} ({len(dir_files)} files)", expanded=(dir_name == "(root)")):
                for f in sorted(dir_files, key=lambda x: x["path"]):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        icon = "📄"
                        if f["content_type"] == "pdf":
                            icon = "📕"
                        elif f["content_type"] == "word":
                            icon = "📘"
                        st.text(f"{icon} {f['path']}")
                    with col2:
                        size_kb = f["size"] / 1024
                        st.text(f"{size_kb:.1f} KB")
                    with col3:
                        st.text(f["content_type"])

        st.divider()

        # File search
        st.subheader("🔍 Search Files")
        search_query = st.text_input("Search for text in files:", key="file_search")
        if search_query:
            search_result = api_request("GET", "/files/search", params={"query": search_query})
            if search_result:
                st.write(f"Found {len(search_result)} files with matches:")
                for match in search_result:
                    with st.expander(f"📄 {match['path']}", expanded=True):
                        for m in match.get("matches", []):
                            st.code(f"Line {m['line']}: {m['content']}")
            else:
                st.info("No matches found")

        st.divider()

        # File preview
        st.subheader("👁️ Preview File")
        file_paths = [f["path"] for f in files_result if f["content_type"] == "text"]
        if file_paths:
            selected_file = st.selectbox("Select a text file to preview:", [""] + file_paths)
            if selected_file:
                response = api_request_raw("GET", "/files/content", params={"path": selected_file})
                if response:
                    content = response.text
                    # Determine language for syntax highlighting
                    if selected_file.endswith(".py"):
                        lang = "python"
                    elif selected_file.endswith(".js") or selected_file.endswith(".ts"):
                        lang = "javascript"
                    elif selected_file.endswith(".json"):
                        lang = "json"
                    elif selected_file.endswith(".md"):
                        lang = "markdown"
                    else:
                        lang = ""

                    if lang:
                        st.code(content[:5000], language=lang)
                    else:
                        st.text(content[:5000])

                    if len(content) > 5000:
                        st.caption(f"... (showing first 5000 chars of {len(content)})")
    else:
        st.info("No files available. Check the whitelist configuration.")

# TAB 3: API Explorer
with tab3:
    st.header("🔧 API Explorer")
    st.caption("Test API endpoints directly")

    endpoint = st.selectbox(
        "Endpoint",
        [
            "GET /health",
            "GET /stats",
            "GET /cache",
            "GET /instructions",
            "GET /model",
            "GET /files",
            "GET /files/content",
            "GET /files/search",
            "GET /backup/list",
            "GET /backup/export",
            "POST /backup/create",
            "GET /threads",
            "POST /recall",
            "POST /remember",
            "POST /chat",
        ]
    )

    if endpoint == "GET /health":
        if st.button("Execute"):
            result = api_request("GET", "/health")
            st.json(result)

    elif endpoint == "GET /stats":
        if st.button("Execute"):
            result = api_request("GET", "/stats")
            st.json(result)

    elif endpoint == "GET /cache":
        if st.button("Execute"):
            result = api_request("GET", "/cache")
            st.json(result)

    elif endpoint == "GET /instructions":
        if st.button("Execute"):
            result = api_request("GET", "/instructions")
            st.json(result)

    elif endpoint == "GET /model":
        if st.button("Execute"):
            result = api_request("GET", "/model")
            st.json(result)

    elif endpoint == "GET /files":
        if st.button("Execute"):
            result = api_request("GET", "/files")
            st.json(result)

    elif endpoint == "GET /files/content":
        path = st.text_input("File path:", placeholder="e.g., sem_mem/core.py")
        if st.button("Execute") and path:
            response = api_request_raw("GET", "/files/content", params={"path": path})
            if response:
                if "application/pdf" in response.headers.get("content-type", ""):
                    st.info("PDF file - binary content")
                    st.download_button("Download PDF", response.content, file_name=path.split("/")[-1])
                elif "application/msword" in response.headers.get("content-type", "") or "wordprocessingml" in response.headers.get("content-type", ""):
                    st.info("Word document - binary content")
                    st.download_button("Download Document", response.content, file_name=path.split("/")[-1])
                else:
                    st.code(response.text[:5000])

    elif endpoint == "GET /files/search":
        query = st.text_input("Search query:")
        max_results = st.slider("Max results", 1, 50, 20)
        if st.button("Execute") and query:
            result = api_request("GET", "/files/search", params={"query": query, "max_results": max_results})
            st.json(result)

    elif endpoint == "GET /backup/list":
        if st.button("Execute"):
            result = api_request("GET", "/backup/list")
            st.json(result)

    elif endpoint == "GET /backup/export":
        include_vectors = st.checkbox("Include vectors", value=True, key="api_export_vectors")
        include_threads = st.checkbox("Include threads", value=True, key="api_export_threads")
        if st.button("Execute"):
            result = api_request("GET", "/backup/export", params={
                "include_vectors": include_vectors,
                "include_threads": include_threads
            })
            st.json(result)

    elif endpoint == "POST /backup/create":
        backup_name = st.text_input("Backup name (optional):", key="api_backup_name")
        if st.button("Execute"):
            params = {}
            if backup_name.strip():
                params["backup_name"] = backup_name.strip()
            result = api_request("POST", "/backup/create", params=params)
            st.json(result)

    elif endpoint == "GET /threads":
        if st.button("Execute"):
            result = api_request("GET", "/threads")
            st.json(result)

    elif endpoint == "POST /recall":
        query = st.text_input("Query:")
        limit = st.slider("Limit", 1, 10, 3)
        threshold = st.slider("Threshold", 0.0, 1.0, 0.40)
        if st.button("Execute") and query:
            result = api_request(
                "POST",
                "/recall",
                json={"query": query, "limit": limit, "threshold": threshold}
            )
            st.json(result)

    elif endpoint == "POST /remember":
        text = st.text_area("Text to remember:")
        if st.button("Execute") and text:
            result = api_request("POST", "/remember", json={"text": text})
            st.json(result)

    elif endpoint == "POST /chat":
        query = st.text_input("Query:")
        if st.button("Execute") and query:
            result = api_request("POST", "/chat", json={"query": query})
            st.json(result)
