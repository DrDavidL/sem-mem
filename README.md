# Sem-Mem: Tiered Semantic Memory for AI Agents

**Sem-Mem** is a local, privacy-first memory layer for OpenAI-based agents. It implements a **Distributed Hash Table (DHT)** architecture using Locality-Sensitive Hashing (LSH) to organize memories by semantic meaning rather than keywords.

It features a **Tiered "Smart Cache"** system (Segmented LRU) that mimics human memory:
1.  **L1 (Hot/RAM):** Instant access to recently used or high-frequency data (Segmented into "Probation" and "Protected" tiers).
2.  **L2 (Cold/Disk):** Permanent storage of guidelines and facts, organized into "Buckets" based on semantic clusters.

## 🚀 Features

* **Zero-Latency "Hot" Recall:** Uses a Segmented LRU cache to keep relevant context in RAM.
* **Privacy-First:** Data stays on your local machine. Only query vectors are sent to OpenAI.
* **Mergeable Minds:** Knowledge is stored in JSON "Buckets" (e.g., `bucket_101.json`). You can share a specific topic bucket with a colleague without sharing your entire database.
* **PDF Ingestion:** "Read" clinical guidelines or papers and auto-chunk them into long-term memory.
* **Memory Atlas:** A 2D visualization (PCA) of your knowledge graph to verify semantic clustering.

## 🛠️ Installation

1.  **Clone or Download** this repository.
2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Run the Chat App
The project includes a Streamlit interface for chatting, training, and visualizing the memory.

```bash
streamlit run app.py
````

### 2\. How to "Teach" the AI

  * **Fact-Based:** Type `Remember: The code for the break room is 1234.` in the chat.
  * **Document-Based:** Use the **Sidebar \> Digest Knowledge** tool to upload a PDF (e.g., a medical guideline).

### 3\. The "Smart Cache" in Action

Watch the Sidebar to see the lifecycle of a memory:

1.  **Ingestion:** New data goes to **L2 (Disk)**.
2.  **First Query:** "What is the protocol for X?" -\> System scans disk (slower) -\> Promotes result to **Probation**.
3.  **Second Query:** "What is the dosage?" -\> System hits **Probation** (Instant).
4.  **Frequent Use:** If used again, the memory moves to **Protected (VIP)**, ensuring it stays in RAM even as you switch topics.

## 📂 Project Structure

```text
.
├── sem_mem/
│   ├── __init__.py
│   └── core.py         # The Logic: LSH Hashing, SmartCache, OpenAI Wrapper
├── local_memory/       # The Data: Auto-generated JSON buckets
├── app.py              # The UI: Streamlit Frontend
├── requirements.txt
└── README.md
```

## 🤝 Collaborative "Mind Merging"

To share knowledge with a colleague using Sem-Mem:

1.  Go to your `local_memory/` folder.
2.  Find the bucket corresponding to the topic (use the "Memory Atlas" in the app to identify the bucket ID).
3.  Send the `bucket_xxxxx.json` file to your colleague.
4.  They use the **"Merge Bucket"** feature in their sidebar to instantly absorb that specific knowledge domain.    