"""
Minimal chatbot example (no Streamlit).

Shows the simplest possible integration with sem_mem.

Run with:
    python examples/minimal_bot.py
"""

import os
from sem_mem import SemanticMemory
from sem_mem.decorators import MemoryChat


def main():
    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter OpenAI API Key: ")

    memory = SemanticMemory(api_key=api_key)
    chat = MemoryChat(memory)

    print("\n=== Minimal Memory Bot ===")
    print("Commands:")
    print("  /remember <fact>  - Save a fact to memory")
    print("  /instruct <text>  - Add a permanent instruction")
    print("  /save             - Save conversation to L2")
    print("  /new              - Start new conversation")
    print("  /quit             - Exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()
            arg = user_input[len(cmd):].strip()

            if cmd == "/quit":
                break
            elif cmd == "/remember" and arg:
                chat.remember(arg)
                print(f"Remembered: {arg}\n")
            elif cmd == "/instruct" and arg:
                chat.add_instruction(arg)
                print(f"Instruction added: {arg}\n")
            elif cmd == "/save":
                count = chat.save_thread()
                print(f"Saved {count} chunks to memory.\n")
            elif cmd == "/new":
                chat.new_thread()
                print("Started new conversation.\n")
            else:
                print("Unknown command.\n")
            continue

        # Regular chat
        response = chat.send(user_input)
        print(f"Bot: {response}\n")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
