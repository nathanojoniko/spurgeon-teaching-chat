import os
import requests
import re
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Fetch Text
url = "https://www.ccel.org/ccel/spurgeon/sermons01.txt"
print("📥 Fetching Volume 1 Text...")
response = requests.get(url)
full_text = response.text

# 3. THE CATCH-ALL SCISSORS
# This splits by "Sermon" (any case) followed by a space and a number/letter
sermon_parts = re.split(r'(?i)sermon\s+', full_text)

# If that still feels too small, we'll know immediately
if len(sermon_parts) < 10:
    print("⚠️ Pattern match low. Falling back to double-newline splitting...")
    sermon_parts = full_text.split('\n\n\n')

print(f"✅ Found {len(sermon_parts)} segments. Starting full index...")

# 4. Push to Pinecone
for i, content in enumerate(sermon_parts):
    if i == 0: continue # Skip intro
    
    # We'll take 20 paragraphs from every single section found
    paragraphs = content.split('\n\n')
    print(f"   📖 Processing Section {i}...")
    
    vectors_to_upsert = []
    for j, para in enumerate(paragraphs[:20]): 
        clean_text = para.strip().replace('\r', '')
        if len(clean_text) > 150:
            vector = model.encode(clean_text).tolist()
            vectors_to_upsert.append({
                "id": f"vol1_sec_{i}_p{j}",
                "values": vector,
                "metadata": {"text": clean_text[:1000], "source": f"Vol 1, Sec {i}"}
            })
            
            if len(vectors_to_upsert) >= 25:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                time.sleep(0.1)

    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

print("\n🚀 SUCCESS: Check Pinecone now. You should see a high 'Total Vector' count!")
