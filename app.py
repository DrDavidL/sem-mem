import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sem_mem.core import SemanticMemory
from sem_mem.config import (
    get_api_key,
    get_config,
    CHAT_MODELS,
    REASONING_EFFORTS,
    AUTO_THREAD_RENAME_ENABLED,
    AUTO_THREAD_RENAME_MIN_USER_MESSAGES,
    CONVERSATION_SUMMARY_TOKEN_THRESHOLD,
    CONVERSATION_SUMMARY_MIN_MESSAGES,
    CONVERSATION_SUMMARY_LEAVE_RECENT,
    CONVERSATION_SUMMARY_MAX_WINDOWS_PER_THREAD,
    ON_DELETE_THREAD_BEHAVIOR,
)
from sem_mem.thread_utils import (
    generate_thread_title,
    estimate_message_tokens,
    select_summary_window,
    summarize_conversation_window,
    summarize_deleted_thread,
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
st.set_page_config(page_title="Semantic Memory Agent", layout="wide")

# --- Initialize Memory Agent ---
if "memory_agent" not in st.session_state:
    api_key = get_api_key()
    if not api_key:
        with st.sidebar:
            api_key = st.text_input("Enter OpenAI API Key", type="password")

    if api_key:
        config = get_config()
        st.session_state.memory_agent = SemanticMemory(
            api_key=api_key,
            storage_dir=config["storage_dir"],
            cache_size=config["cache_size"],
            embedding_model=config["embedding_model"],
            chat_model=config["chat_model"],
            reasoning_effort=config["reasoning_effort"],
            auto_memory=False,  # Streamlit UI manages its own memory operations
        )

if "memory_agent" not in st.session_state:
    st.warning("Please provide an API Key to start.")
    st.stop()

agent = st.session_state.memory_agent

# Initialize model settings in session state
if "chat_model" not in st.session_state:
    st.session_state.chat_model = agent.chat_model
if "reasoning_effort" not in st.session_state:
    st.session_state.reasoning_effort = agent.reasoning_effort

# --- Thread Management ---
# Thread structure:
#   {
#       "messages": [],              # List of {role, content} dicts
#       "response_id": None,         # OpenAI Responses API ID for continuity
#       "title": "New conversation", # Display title (auto-generated or user-set)
#       "title_user_overridden": False,  # If True, auto-rename won't touch it
#       "summary_windows": [],       # List of summary window dicts (see below)
#   }
#
# Summary window structure:
#   {
#       "start_index": 0,      # Start of summarized message range (inclusive)
#       "end_index": 24,       # End of summarized message range (exclusive)
#       "summary_text": "...", # The generated summary
#       "summary_id": "...",   # L2 memory ID (or None if storage failed)
#       "timestamp": "...",    # ISO timestamp when summary was created
#   }


def _create_empty_thread() -> dict:
    """Create a new empty thread with default structure."""
    return {
        "messages": [],
        "response_id": None,
        "title": "New conversation",
        "title_user_overridden": False,
        "summary_windows": [],
    }


if "threads" not in st.session_state:
    st.session_state.threads = {"Thread 1": _create_empty_thread()}
if "current_thread" not in st.session_state:
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


def _store_summary_in_memory(
    summary_text: str,
    thread_name: str,
    start_index: int,
    end_index: int,
) -> str | None:
    """
    Store a conversation summary in L2 memory.

    The summary text is embedded for semantic retrieval, and metadata
    is stored alongside to track which thread/range it came from.

    Args:
        summary_text: The generated summary content
        thread_name: Name/ID of the thread
        start_index: Start of the summarized message range
        end_index: End of the summarized message range

    Returns:
        Result message from memory.remember(), or None on failure
    """
    if not summary_text:
        return None

    # Build a structured payload that embeds well but preserves metadata
    # The summary text itself is what gets embedded; metadata is stored alongside
    metadata = {
        "type": "conversation_summary",
        "thread_name": thread_name,
        "start_index": start_index,
        "end_index": end_index,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        result = agent.remember(summary_text, metadata=metadata)
        return result
    except Exception:
        # Fail soft; summarization is opportunistic
        return None


def _maybe_summarize_thread(thread_data: dict, thread_name: str) -> bool:
    """
    Check if thread needs summarization and create a summary window if so.

    Args:
        thread_data: The thread dict with messages and summary_windows
        thread_name: Name of the thread (for metadata)

    Returns:
        True if a summary was created, False otherwise
    """
    messages = thread_data.get("messages", [])
    windows = thread_data.get("summary_windows", [])

    # Check basic conditions
    if len(messages) < CONVERSATION_SUMMARY_MIN_MESSAGES:
        return False

    if len(windows) >= CONVERSATION_SUMMARY_MAX_WINDOWS_PER_THREAD:
        return False

    # Check token threshold
    approx_tokens = estimate_message_tokens(messages)
    if approx_tokens < CONVERSATION_SUMMARY_TOKEN_THRESHOLD:
        return False

    # Select window to summarize
    window = select_summary_window(
        messages,
        windows,
        leave_recent=CONVERSATION_SUMMARY_LEAVE_RECENT,
        min_messages=CONVERSATION_SUMMARY_MIN_MESSAGES,
    )

    if window is None:
        return False

    start_idx, end_idx = window
    window_messages = messages[start_idx:end_idx]

    # Generate summary
    summary_text = summarize_conversation_window(
        window_messages,
        client=agent.client,
    )

    if not summary_text:
        return False

    # Store in L2
    summary_id = _store_summary_in_memory(
        summary_text,
        thread_name=thread_name,
        start_index=start_idx,
        end_index=end_idx,
    )

    # Add to thread's summary windows
    windows.append({
        "start_index": start_idx,
        "end_index": end_idx,
        "summary_text": summary_text,
        "summary_id": summary_id,
        "timestamp": datetime.utcnow().isoformat(),
    })
    thread_data["summary_windows"] = windows

    return True


def _store_farewell_summary_in_memory(
    summary_text: str,
    thread_name: str,
    thread_data: dict,
) -> str | None:
    """
    Store a farewell summary (from thread deletion) in L2 memory.

    Similar to _store_summary_in_memory but for entire-thread summaries
    created when a thread is deleted.

    Args:
        summary_text: The generated farewell summary
        thread_name: Name/ID of the thread being deleted
        thread_data: The full thread dict (for metadata like message count)

    Returns:
        Result message from memory.remember(), or None on failure
    """
    if not summary_text:
        return None

    message_count = len(thread_data.get("messages", []))
    thread_title = thread_data.get("title", thread_name)

    metadata = {
        "type": "farewell_summary",
        "thread_name": thread_name,
        "thread_title": thread_title,
        "message_count": message_count,
        "timestamp": datetime.utcnow().isoformat(),
    }

    try:
        result = agent.remember(summary_text, metadata=metadata)
        return result
    except Exception:
        # Fail soft; summarization is opportunistic
        return None


def _handle_delete_thread(thread_name: str, save_summary: bool) -> bool:
    """
    Delete a thread, optionally saving a farewell summary first.

    Args:
        thread_name: Name of the thread to delete
        save_summary: Whether to generate and save a farewell summary

    Returns:
        True if deletion succeeded, False otherwise
    """
    if thread_name not in st.session_state.threads:
        return False

    thread_data = st.session_state.threads[thread_name]

    # Generate and store farewell summary if requested
    if save_summary:
        messages = thread_data.get("messages", [])
        if len(messages) >= 2:  # Only summarize non-trivial threads
            summary = summarize_deleted_thread(messages, client=agent.client)
            if summary:
                _store_farewell_summary_in_memory(summary, thread_name, thread_data)

    # Delete the thread
    del st.session_state.threads[thread_name]

    # Switch to another thread or create a new one
    remaining_threads = list(st.session_state.threads.keys())
    if remaining_threads:
        st.session_state.current_thread = remaining_threads[0]
    else:
        # No threads left, create a new one
        st.session_state.threads["Thread 1"] = _create_empty_thread()
        st.session_state.current_thread = "Thread 1"

    return True


# --- Sidebar: The "File System" View ---
with st.sidebar:
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
            # "never_save" or thread is too short to summarize
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
            st.rerun()
    with col2:
        current_messages = st.session_state.threads[st.session_state.current_thread]["messages"]
        if st.button("Save to L2", disabled=len(current_messages) == 0):
            with st.spinner("Saving thread to memory..."):
                count = agent.save_thread_to_memory(
                    current_messages,
                    thread_name=st.session_state.current_thread
                )
            st.success(f"Saved {count} chunks to L2!")

    st.divider()

    # Model Selection
    st.header("🤖 Model")
    model_options = list(CHAT_MODELS.keys())
    selected_model = st.selectbox(
        "Chat Model",
        model_options,
        index=model_options.index(st.session_state.chat_model),
        help="gpt-5.1 is a reasoning model (like o3)"
    )
    if selected_model != st.session_state.chat_model:
        st.session_state.chat_model = selected_model
        agent.chat_model = selected_model

    # Show reasoning effort only for reasoning models
    if selected_model in ("gpt-5.1", "o1", "o3"):
        selected_effort = st.select_slider(
            "Reasoning Effort",
            options=REASONING_EFFORTS,
            value=st.session_state.reasoning_effort,
            help="Higher = more thorough but slower"
        )
        if selected_effort != st.session_state.reasoning_effort:
            st.session_state.reasoning_effort = selected_effort
            agent.reasoning_effort = selected_effort

    # Web Search Toggle
    web_search_enabled = st.toggle(
        "🌐 Web Search",
        value=st.session_state.get("web_search", False),
        help="Enable web search for real-time information"
    )
    if web_search_enabled != st.session_state.get("web_search", False):
        st.session_state.web_search = web_search_enabled
        agent.web_search_enabled = web_search_enabled

    st.divider()

    # Instructions Editor
    st.header("📋 Instructions")
    st.caption("Permanent context included in every request:")
    current_instructions = agent.load_instructions()
    new_instructions = st.text_area(
        "Edit instructions",
        value=current_instructions,
        height=100,
        label_visibility="collapsed"
    )
    if st.button("Save Instructions"):
        agent.save_instructions(new_instructions)
        st.success("Instructions saved!")

    st.divider()
    st.header("🧠 Distributed Memory")

    # 1. PDF Ingestion
    with st.expander("📚 Digest Knowledge (PDF)", expanded=True):
        uploaded_pdf = st.file_uploader("Upload a Guideline", type="pdf")
        if uploaded_pdf and st.button("Ingest PDF"):
            with st.spinner("Reading, Chunking, and Memorizing..."):
                count = agent.bulk_learn_pdf(uploaded_pdf)
                st.success(f"Absorbed {count} new concepts!")
                st.rerun()

    # 2. JSON Merger
    with st.expander("📥 Merge Minds (JSON)"):
        uploaded_json = st.file_uploader("Drop memories.json", type="json")
        if uploaded_json and st.button("Import Memories"):
            import json
            entries = json.load(uploaded_json)
            count = agent.import_memories(entries)
            st.success(f"Imported {count} new memories!")

    st.divider()
    
    # 3. Hot Cache Monitor (NEW)
    st.subheader("🔥 Hot Cache (RAM)")
    st.caption("Memories currently ready for instant access:")
    if hasattr(agent, 'local_cache'):
        cache_items = list(agent.local_cache)
        if not cache_items:
            st.info("Cache is empty. Ask something!")
        else:
            for i, item in enumerate(cache_items):
                st.text(f"{i+1}. {item['text'][:40]}...")
    
    st.divider()

    # 4. Global Stats - using HNSW index
    stats = agent.get_stats()
    st.metric("Total L2 Memories", stats["l2_memories"])
    st.metric("L1 Cache Size", stats["l1_cache_size"])

    # Get all memories for visualization
    all_memories = []
    for entry in agent.vector_index.get_all_entries():
        all_memories.append({
            "text": entry['text'],
            "vector": entry['vector'],
            "source": entry.get('metadata', {}).get('source', 'unknown'),
        })

    st.divider()

    # 5. Sema File Access Management
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
        # Manual entry
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

        # Suggestions
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
st.title("Sem-Mem: The Tiered Memory Agent")

# Multi-session info
with st.expander("ℹ️ Multi-Session Info", expanded=False):
    st.info(
        "**Standalone Mode**: L2 memories (disk) are shared across sessions, "
        "but L1 cache (RAM) is isolated per browser tab.\n\n"
        "For fully shared memory across multiple users, use the **API server**:\n"
        "```\nuvicorn server:app --reload\nstreamlit run app_api.py\n```"
    )

tab1, tab2 = st.tabs(["💬 Chat & Learn", "🗺️ Memory Atlas"])

# TAB 1: Chat
with tab1:
    thread_data = st.session_state.threads[st.session_state.current_thread]
    messages = thread_data["messages"]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question, 'Remember: ...', or 'Instruct: ...'"):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. Instruct Command - Add to permanent instructions
        if prompt.lower().startswith("instruct:"):
            instruction = prompt.split(":", 1)[1].strip()
            agent.add_instruction(instruction)
            response = f"✅ **Instruction added.**\n*\"{instruction}\"*"
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        # 2. Remember Command - Add to semantic memory (L2)
        elif prompt.lower().startswith("remember:"):
            fact = prompt.split(":", 1)[1].strip()
            bucket_msg = agent.remember(fact)
            response = f"✅ **Stored.** \n*({bucket_msg})*"
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        # 3. Chat with RAG
        else:
            prev_id = thread_data["response_id"]
            response_text, new_response_id, retrieved_mems, logs = agent.chat_with_memory(
                prompt, previous_response_id=prev_id
            )
            thread_data["response_id"] = new_response_id

            # Formatting Response
            full_response = response_text

            # Append Retrieval Logs
            if logs:
                log_str = "\n".join([f"`{l}`" for l in logs])
                full_response += f"\n\n**System Logs:**\n{log_str}"

            # Append Citations
            if retrieved_mems:
                sources = "\n\n**Retrieved Context:**\n" + "\n".join([f"> *{m}*" for m in retrieved_mems])
                full_response += sources

            messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)

        # Auto-rename thread if eligible (runs after any chat interaction)
        if _should_auto_rename(thread_data):
            new_title = generate_thread_title(messages, client=agent.client)
            if new_title:
                thread_data["title"] = new_title.strip()
                st.rerun()  # Refresh to show new title in sidebar

        # Auto-summarize thread if it's getting long (opportunistic)
        # This creates "chapter" summaries stored in L2 for durable semantic history
        if _maybe_summarize_thread(thread_data, st.session_state.current_thread):
            st.toast("Created summary of earlier conversation", icon="📝")

# TAB 2: Atlas
with tab2:
    st.header("Semantic Space Visualization")
    if len(all_memories) < 3:
        st.info("Need at least 3 memories to generate a graph.")
    else:
        df = pd.DataFrame(all_memories)
        vectors = np.array(df['vector'].tolist())
        pca = PCA(n_components=2)
        components = pca.fit_transform(vectors)
        df['x'] = components[:, 0]
        df['y'] = components[:, 1]

        fig = px.scatter(
            df, x='x', y='y', color='source',
            hover_data=['text'], title="Memory Clusters (HNSW Index)",
            template="plotly_dark", size_max=15
        )
        st.plotly_chart(fig, use_container_width=True)