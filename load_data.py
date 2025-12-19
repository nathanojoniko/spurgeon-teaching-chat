import os
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup (Same as before)
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. The STABLE Link (Morning & Evening by Spurgeon)
# This is a public, plain-text link from CCEL
url = "https://www.ccel.org/ccel/spurgeon/morneve.txt"

print("📥 Downloading Spurgeon's Morning & Evening...")
response = requests.get(url)
text = response.text

# 3. Clean and Chunk
# We split by '***' because CCEL uses that to separate days/sections
sections = text.split('***')
print(f"✅ Found {len(sections)} sections. Loading the first 50...")

# 4. Push to Pinecone
for i, section in enumerate(sections[10:60]): # Skip the first few intro lines
    clean_text = section.strip()
    if len(clean_text) > 100:
        vector = model.encode(clean_text).tolist()
        index.upsert(vectors=[{
            "id": f"morneve_{i}",
            "values": vector,
            "metadata": {"text": clean_text[:1000], "source": "Morning & Evening"}
        }])

print("🎉 DONE! Your library is now officially stocked.")
