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

# 2. Fetch the ENTIRE Volume 1 (Sermons 1-53)
url = "https://www.ccel.org/ccel/spurgeon/sermons01.txt"
print("📥 Fetching the complete Volume 1...")
response = requests.get(url)
full_text = response.text

# 3. Split by Sermon headers
# This handles the Roman numerals and digits CCEL uses
sermon_parts = re.split(r'(?i)sermon\s+[0-9ivxl]+[\s.]', full_text)

print(f"✅ Found {len(sermon_parts)-1} sermons total. Starting full index...")

# 4. Process Every Sermon
for i, sermon_content in enumerate(sermon_parts):
    if i == 0: continue # Skip the Table of Contents/Preface
    
    sermon_content = sermon_content.strip()
    paragraphs = sermon_content.split('\n\n')
    
    print(f"   📖 Indexing Sermon {i} ({len(paragraphs)} paragraphs)...")
    
    vectors_to_upsert = []
    for j, para in enumerate(paragraphs):
        clean_text = para.strip().replace('\r', '')
        
        # Only index meaningful chunks
        if len(clean_text) > 150:
            vector = model.encode(clean_text).tolist()
            vectors_to_upsert.append({
                "id": f"vol1_s_{i}_p{j}",
                "values": vector,
                "metadata": {
                    "text": clean_text[:1000], # AI reads this
                    "sermon_num": i,
                    "source": "Sermons Vol 1"
                }
            })
            
            # Upsert in batches of 25 to be safe and efficient
            if len(vectors_to_upsert) >= 25:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
                time.sleep(0.1) # Small pause to stay under free tier rate limits

    # Catch any leftovers for the current sermon
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)

print("\n🚀 MISSION ACCOMPLISHED: The entire first volume is now live in your library!")
