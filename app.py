import streamlit as st
import os
import glob
import json
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sem_mem.core import SemanticMemory

# Page Config
st.set_page_config(page_title="Semantic Memory Agent", layout="wide")

# --- Initialize Memory Agent ---
if "memory_agent" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        with st.sidebar:
            api_key = st.text_input("Enter OpenAI API Key", type="password")
    
    if api_key:
        st.session_state.memory_agent = SemanticMemory(api_key=api_key)

if "memory_agent" not in st.session_state:
    st.warning("Please provide an API Key to start.")
    st.stop()

agent = st.session_state.memory_agent

# --- Sidebar: The "File System" View ---
with st.sidebar:
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

    # 4. Global Stats
    bucket_files = glob.glob("./local_memory/*.json")
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

tab1, tab2 = st.tabs(["💬 Chat & Learn", "🗺️ Memory Atlas"])

# TAB 1: Chat
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question or type 'Remember: ...'"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 1. Direct Command
        if prompt.lower().startswith("remember:"):
            fact = prompt.split(":", 1)[1].strip()
            bucket_msg = agent.remember(fact)
            response = f"✅ **Stored.** \n*({bucket_msg})*"
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            st.rerun() # Refresh sidebar
            
        # 2. Chat with RAG
        else:
            response_text, retrieved_mems, logs = agent.chat_with_memory(prompt)
            
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

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            with st.chat_message("assistant"):
                st.markdown(full_response)
            st.rerun() # Refresh sidebar (Cache updates)

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
        st.plotly_chart(fig, use_container_width=True)