"""
Streamlit chat app using the decorator-based API.

This demonstrates @with_memory for adding memory to custom chat functions.

Run with:
    streamlit run examples/decorator_chat.py
"""

import streamlit as st
import os

from sem_mem import SemanticMemory, with_memory

st.set_page_config(page_title="Decorator-Based Chat", layout="centered")
st.title("Decorator-Based Chat")

# --- Initialize Memory ---
if "memory" not in st.session_state:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if not api_key:
            st.stop()

    st.session_state.memory = SemanticMemory(api_key=api_key)
    st.session_state.messages = []
    st.session_state.response_id = None

memory = st.session_state.memory


# --- Define chat function with memory decorator ---
@with_memory(memory, auto_remember=False)
def chat_with_context(
    user_input: str,
    context: str = "",
    instructions: str = "",
    previous_response_id: str = None,
    _set_response_id=None,
    **kwargs,
) -> str:
    """
    Custom chat function that receives injected memory context.

    The @with_memory decorator automatically:
    - Retrieves relevant memories (context)
    - Loads instructions from instructions.txt
    - Tracks conversation state (previous_response_id)
    """
    # Build input with context
    if context:
        input_text = f"{context}\n\nUser: {user_input}"
    else:
        input_text = user_input

    # Call the API
    response = memory.client.responses.create(
        model="gpt-4o",
        instructions=instructions,
        input=input_text,
        **({"previous_response_id": previous_response_id} if previous_response_id else {}),
    )

    # Update conversation state
    if _set_response_id:
        _set_response_id(response.id)

    # Store for next call
    st.session_state.response_id = response.id

    return response.output_text


# --- Sidebar ---
with st.sidebar:
    st.header("Memory Info")

    # Show current instructions
    instructions = memory.load_instructions()
    st.text_area("Current Instructions", value=instructions or "(none)", disabled=True)

    # Add instruction
    new_instruction = st.text_input("Add instruction")
    if st.button("Add") and new_instruction:
        memory.add_instruction(new_instruction)
        st.success("Added!")
        st.rerun()

    st.divider()

    # Remember fact
    fact = st.text_input("Remember a fact")
    if st.button("Remember") and fact:
        memory.remember(fact)
        st.success("Remembered!")

    st.divider()

    # Clear conversation
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.response_id = None
        st.rerun()

# --- Chat Interface ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # The decorator handles memory retrieval and context injection
            response = chat_with_context(
                prompt,
                previous_response_id=st.session_state.response_id,
            )
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
