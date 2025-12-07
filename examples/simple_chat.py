"""
Simple Streamlit chat app using the sem_mem package.

This demonstrates the MemoryChat class for easy integration.

Run with:
    streamlit run examples/simple_chat.py
"""

import streamlit as st
import os

from sem_mem import SemanticMemory
from sem_mem.decorators import MemoryChat

st.set_page_config(page_title="Simple Memory Chat", layout="centered")
st.title("Simple Memory Chat")

# --- Initialize ---
if "chat" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if not api_key:
            st.stop()

    memory = SemanticMemory(api_key=api_key)
    st.session_state.chat = MemoryChat(memory)

chat = st.session_state.chat

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Memory Controls")

    # Instructions editor
    st.subheader("Instructions")
    instructions = st.text_area(
        "System instructions",
        value=chat.instructions,
        height=100,
    )
    if st.button("Save Instructions"):
        chat.memory.save_instructions(instructions)
        st.success("Saved!")

    st.divider()

    # Thread controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Thread"):
            chat.new_thread()
            st.rerun()
    with col2:
        if st.button("Save Thread", disabled=len(chat.messages) == 0):
            count = chat.save_thread()
            st.success(f"Saved {count} chunks!")

    st.divider()

    # Quick remember
    st.subheader("Quick Remember")
    fact = st.text_input("Fact to remember")
    if st.button("Remember", disabled=not fact):
        chat.remember(fact)
        st.success("Remembered!")

# --- Chat Interface ---
for msg in chat.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat.send(prompt)
        st.markdown(response)
