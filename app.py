import streamlit as st
import os
import glob
import json
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sem_mem.core import SemanticMemory
from sem_mem.config import get_api_key, get_config, CHAT_MODELS, REASONING_EFFORTS

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
if "threads" not in st.session_state:
    st.session_state.threads = {"Thread 1": {"messages": [], "response_id": None}}
if "current_thread" not in st.session_state:
    st.session_state.current_thread = "Thread 1"

# --- Sidebar: The "File System" View ---
with st.sidebar:
    # Thread Controls
    st.header("💬 Threads")
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
        if st.button("💾 Save to L2", disabled=len(current_messages) == 0):
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
        uploaded_json = st.file_uploader("Drop bucket_xxx.json", type="json")
        if uploaded_json and st.button("Merge Bucket"):
            msg = agent.import_bucket(uploaded_json)
            st.success(msg)

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

    # 4. Global Stats - use agent.storage_dir instead of hardcoded path
    bucket_pattern = os.path.join(agent.storage_dir, "bucket_*.json")
    bucket_files = glob.glob(bucket_pattern)
    total_memories = 0
    all_memories = []

    for f in bucket_files:
        with open(f, 'r') as file:
            data = json.load(file)
            total_memories += len(data)
            for item in data:
                all_memories.append({
                    "text": item['text'],
                    "bucket": os.path.basename(f).replace(".json", ""),
                    "vector": item['vector']
                })

    st.metric("Total L2 Memories", total_memories)
    st.metric("Active Buckets", len(bucket_files))

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
            df, x='x', y='y', color='bucket', 
            hover_data=['text'], title="Memory Clusters (LSH Buckets)",
            template="plotly_dark", size_max=15
        )
        st.plotly_chart(fig, width="stretch")