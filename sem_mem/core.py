import os
import json
import numpy as np
from openai import OpenAI
from collections import deque, OrderedDict
from pypdf import PdfReader
from typing import List, Dict

class SmartCache:
    def __init__(self, capacity=20, protected_ratio=0.8):
        self.capacity = capacity
        # Split memory into two segments
        self.protected_cap = int(capacity * protected_ratio)
        self.probation_cap = capacity - self.protected_cap
        
        # OrderedDict allows us to move items to 'end' (Newest) easily
        self.protected = OrderedDict()
        self.probation = OrderedDict()

    def get(self, key_text):
        """
        Retrieves an item and updates its status (The "Reset" Logic).
        """
        # 1. Check Protected (VIP)
        if key_text in self.protected:
            # HIT in Protected: "Reset" it to the newest position
            item = self.protected.pop(key_text)
            self.protected[key_text] = item
            return item, "Protected"

        # 2. Check Probation (New/Transient)
        if key_text in self.probation:
            # HIT in Probation: PROMOTE to Protected
            item = self.probation.pop(key_text)
            self._add_to_protected(key_text, item)
            return item, "Promoted to Protected"
            
        return None, "Miss"

    def add(self, item):
        """
        New items always start in Probation.
        """
        key = item['text']
        # If it's already known, just refresh it
        if key in self.protected or key in self.probation:
            self.get(key)
            return

        # Add to Probation (Newest)
        self.probation[key] = item
        
        # If Probation is full, evict the oldest (FIFO behavior for new stuff)
        if len(self.probation) > self.probation_cap:
            self.probation.popitem(last=False) # last=False pops the OLDEST

    def _add_to_protected(self, key, item):
        """
        Handles the logic of adding to the VIP section.
        """
        self.protected[key] = item
        
        # If Protected is full, we don't delete. We DEMOTE the oldest VIP back to Probation.
        # This gives it one last chance to be used before dying.
        if len(self.protected) > self.protected_cap:
            demoted_key, demoted_val = self.protected.popitem(last=False)
            self.add(demoted_val) # Send back to Probation

    def list_items(self):
        """Helper to visualize the segments."""
        return {
            "Protected (Sticky)": list(reversed(self.protected.values())),
            "Probation (Transient)": list(reversed(self.probation.values()))
        }

class SemanticMemory:
    def __init__(self, api_key=None, storage_dir="./local_memory", cache_size=20):
        self.client = OpenAI(api_key=api_key)
        self.storage_dir = storage_dir
        
        # REPLACE deque with SmartCache
        self.local_cache = SmartCache(capacity=cache_size)
        
        # (Rest of init remains the same...)
        np.random.seed(42) 
        self.projection_matrix = np.random.randn(1536, 5)
        if not os.path.exists(storage_dir): os.makedirs(storage_dir)

    def _get_embedding(self, text):
        resp = self.client.embeddings.create(input=text, model="text-embedding-3-small")
        return np.array(resp.data[0].embedding)

    def _lsh_hash(self, vector):
        """Project high-dim vector to low-dim binary string."""
        dot_product = np.dot(vector, self.projection_matrix)
        binary_str = "".join(['1' if x > 0 else '0' for x in dot_product])
        return binary_str

    def remember(self, text: str, metadata: Dict = None):
        """
        Write-Through Strategy: Saves to Disk (L2) AND promotes to RAM (L1).
        """
        vector = self._get_embedding(text)
        bucket_id = self._lsh_hash(vector)
        filename = os.path.join(self.storage_dir, f"bucket_{bucket_id}.json")

        entry = {
            "text": text,
            "vector": vector.tolist(),
            "metadata": metadata or {}
        }

        # 1. Save to L2 (Disk)
        current_data = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                current_data = json.load(f)
        
        # Simple deduplication
        if not any(d['text'] == text for d in current_data):
            current_data.append(entry)
            with open(filename, 'w') as f:
                json.dump(current_data, f)
            msg = f"Persisted to bucket_{bucket_id}.json"
        else:
            msg = "Memory already exists on disk."

        # 2. Promote to L1 (RAM)
        # If we just learned it, we assume it's 'Hot'
        if entry not in self.local_cache:
            self.local_cache.appendleft(entry)
            msg += " & Promoted to Hot Cache."
            
        return msg

    def recall(self, query: str, limit=3, threshold=0.82):
            query_vec = self._get_embedding(query)
            logs = []

            # --- TIER 1: Check Smart Cache ---
            # We scan ALL items in both segments of the cache
            all_cache_items = list(self.local_cache.protected.values()) + list(self.local_cache.probation.values())
            
            l1_hits = []
            for item in all_cache_items:
                score = np.dot(query_vec, np.array(item['vector']))
                if score > threshold:
                    l1_hits.append((score, item))
            
            if l1_hits:
                l1_hits.sort(key=lambda x: x[0], reverse=True)
                best_hit = l1_hits[0][1]
                
                # CRITICAL: Trigger the "Reset/Promote" logic on the SmartCache
                # We "get" it to tell the cache "This was used!"
                _, status = self.local_cache.get(best_hit['text'])
                
                logs.append(f"⚡ L1 HIT ({status}) | Conf: {l1_hits[0][0]:.2f}")
                return [x[1]['text'] for x in l1_hits[:limit]], logs

            # --- TIER 2: Check L2 (Disk) ---
            logs.append("🐢 L1 Miss... Searching Disk...")
            bucket_id = self._lsh_hash(query_vec)
            filename = os.path.join(self.storage_dir, f"bucket_{bucket_id}.json")

            if not os.path.exists(filename): return [], logs

            with open(filename, 'r') as f: data = json.load(f)

            l2_results = []
            for item in data:
                score = np.dot(query_vec, np.array(item['vector']))
                l2_results.append((score, item))
                
            l2_results.sort(key=lambda x: x[0], reverse=True)
            top_results = l2_results[:limit]
            
            # --- PROMOTION ---
            promoted = 0
            for score, item in top_results:
                if score > threshold:
                    # Add to Smart Cache (Starts in Probation)
                    self.local_cache.add(item)
                    promoted += 1
            
            if promoted: logs.append(f"🔼 Promoted {promoted} to Probation.")

            return [x[1]['text'] for x in top_results], logs

    def bulk_learn_pdf(self, pdf_file, chunk_size=500):
        """Reads PDF, chunks it, and saves to L2."""
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
        
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            if len(chunk.strip()) > 20: 
                chunks.append(chunk)
        
        count = 0
        for c in chunks:
            self.remember(c, metadata={"source": "pdf_upload"})
            count += 1
        return count

    def import_bucket(self, uploaded_file):
        """Merges an external JSON bucket."""
        new_data = json.load(uploaded_file)
        if not new_data: return "Empty bucket."
        
        # Re-hash to verify destination
        sample_vec = np.array(new_data[0]['vector'])
        bucket_id = self._lsh_hash(sample_vec)
        target_file = os.path.join(self.storage_dir, f"bucket_{bucket_id}.json")

        existing_data = []
        if os.path.exists(target_file):
            with open(target_file, 'r') as f:
                existing_data = json.load(f)
        
        # Merge
        existing_texts = {d['text'] for d in existing_data}
        added_count = 0
        for entry in new_data:
            if entry['text'] not in existing_texts:
                existing_data.append(entry)
                added_count += 1
        
        with open(target_file, 'w') as f:
            json.dump(existing_data, f)
            
        return f"Merged {added_count} items into bucket_{bucket_id}."

    def chat_with_memory(self, user_query, system_prompt="You are a helpful assistant."):
        """Agentic Wrapper."""
        memories, logs = self.recall(user_query)
        context_block = "\n".join([f"- {m}" for m in memories])
        
        full_system = f"{system_prompt}\n\nRELEVANT MEMORY:\n{context_block}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_query}
            ]
        )
        return response.choices[0].message.content, memories, logs