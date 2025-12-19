import os
import requests
import re
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Direct Link to CCEL Plain Text for Volume 1
url = "https://www.ccel.org/ccel/spurgeon/sermons01.txt"

print("📥 Fetching Volume 1 Text...")
response = requests.get(url)
full_text = response.text

# 3. Split by Sermon headers
# This looks for "Sermon 1.", "Sermon 2.", etc.
sermon_parts = re.split(r'Sermon\s+\d+\.', full_text)

# The first part [0] is usually the table of contents/preface.
# We'll loop through sermon_parts[1] to sermon_parts[10].
print(f"✅ Found {len(sermon_parts)-1} sermons. Indexing the first 10...")

for i in range(1, 11):
    if i < len(sermon_parts):
        sermon_content = sermon_parts[i].strip()
        
        # Split the sermon into paragraphs to make them searchable
        paragraphs = sermon_content.split('\n\n')
        
        print(f"   📖 Indexing Sermon {i}...")
        
        for j, para in enumerate(paragraphs):
            clean_text = para.strip()
            # Only index paragraphs that are meaningful length
            if len(clean_text) > 100:
                vector = model.encode(clean_text).tolist()
                index.upsert(vectors=[{
                    "id": f"vol1_sermon_{i}_p{j}",
                    "values": vector,
                    "metadata": {
                        "text": clean_text[:1000], # Store text in metadata for the AI to read
                        "sermon": i,
                        "source": "Sermons Vol 1"
                    }
                }])

print("\n🚀 SUCCESS: First 10 sermons are loaded into Pinecone!")
