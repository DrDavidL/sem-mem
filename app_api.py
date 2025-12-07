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
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(page_title="Sem-Mem (API Client)", layout="wide")

# --- API Configuration ---
API_URL = os.getenv("SEMMEM_API_URL", "http://localhost:8000")


def api_request(method: str, endpoint: str, **kwargs) -> dict:
    """Make a request to the API server."""
    url = f"{API_URL}{endpoint}"
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
    except httpx.ConnectError:
        st.error(f"Cannot connect to API server at {API_URL}. Is it running?")
        st.stop()
    except httpx.HTTPStatusError as e:
        st.error(f"API error: {e.response.status_code} - {e.response.text}")
        return {}


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

# --- Thread Management (client-side only) ---
if "threads" not in st.session_state:
    st.session_state.threads = {"Thread 1": {"messages": [], "response_id": None}}
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "Thread 1"

# --- Sidebar ---
with st.sidebar:
    st.caption(f"Connected to: `{API_URL}`")

    # Thread Controls
    st.header("Threads")
    thread_names = list(st.session_state.threads.keys())
    selected_thread = st.selectbox(
        "Current Thread",
        thread_names,
        index=thread_names.index(st.session_state.current_thread)
    )
    if selected_thread != st.session_state.current_thread:
        st.session_state.current_thread = selected_thread
        st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ New Thread"):
            new_id = len(st.session_state.threads) + 1
            new_name = f"Thread {new_id}"
            st.session_state.threads[new_name] = {"messages": [], "response_id": None}
            st.session_state.current_thread = new_name
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
        is_reasoning = model_config.get("is_reasoning_model", True)

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
    st.header("Distributed Memory")

    # PDF Upload
    with st.expander("Digest Knowledge (PDF)", expanded=True):
        uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_pdf and st.button("Ingest PDF"):
            with st.spinner("Processing PDF..."):
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                result = api_request("POST", "/upload/pdf", files=files)
            if result:
                st.success(f"Processed {result.get('pages', 0)} pages, stored {result.get('chunks_stored', 0)} chunks!")
                st.cache_data.clear()

    # Batch Remember
    with st.expander("Batch Remember"):
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
    st.subheader("Hot Cache (L1)")
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

# --- Main Interface ---
st.title("Sem-Mem: API Client")

tab1, tab2, tab3 = st.tabs(["Chat", "Memory Atlas", "API Explorer"])

# TAB 1: Chat
with tab1:
    thread_data = st.session_state.threads[st.session_state.current_thread]
    messages = thread_data["messages"]

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question, 'remember: ...', or 'instruct: ...'"):
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. Instruct Command
        if prompt.lower().startswith("instruct:"):
            instruction = prompt.split(":", 1)[1].strip()
            api_request("POST", "/instructions", json={"instruction": instruction})
            response = f"**Instruction added.**\n*\"{instruction}\"*"
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.cache_data.clear()

        # 2. Remember Command
        elif prompt.lower().startswith("remember:"):
            fact = prompt.split(":", 1)[1].strip()
            result = api_request("POST", "/remember", json={"text": fact})
            response = f"**Stored.** \n*({result.get('message', '')})*"
            messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.cache_data.clear()

        # 3. Chat with RAG
        else:
            prev_id = thread_data["response_id"]
            with st.spinner("Thinking..."):
                # Get current model config for the request
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
                with st.chat_message("assistant"):
                    st.markdown(full_response)

# TAB 2: Atlas
with tab2:
    st.header("Semantic Space Visualization")
    st.info("Memory Atlas visualization requires direct index access. Use the standalone app for full visualization.")

    # Show stats instead
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Memories (HNSW)", stats.get("total_l2_memories", 0))
        with col2:
            st.metric("Cache Items (L1)", stats.get("l1_cache_size", 0))

# TAB 3: API Explorer
with tab3:
    st.header("API Explorer")
    st.caption("Test API endpoints directly")

    endpoint = st.selectbox(
        "Endpoint",
        [
            "GET /health",
            "GET /stats",
            "GET /cache",
            "GET /instructions",
            "GET /model",
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
