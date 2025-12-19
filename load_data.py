import os
import requests
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat"
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. GitHub API - Target the Root
REPO_API_URL = "https://api.github.com/repos/lyteword/chspurgeon-sermons/contents/"

print("📥 Fetching Volume list...")
response = requests.get(REPO_API_URL)
repo_contents = response.json()

# Filter for the Volume folders (volume-1, volume-2, etc.)
volume_folders = [item for item in repo_contents if item['type'] == 'dir' and 'volume-' in item['name']]

print(f"✅ Found {len(volume_folders)} volumes. Sampling 4 sermons from each of the first 5 volumes...")

# 3. Process first 5 Volumes, 4 sermons each
total_sermon_count = 0

for vol in volume_folders[:5]: # Look at first 5 volumes
    vol_name = vol['name']
    vol_url = vol['url']
    
    print(f"📂 Entering {vol_name}...")
    vol_response = requests.get(vol_url)
    sermon_list = vol_response.json()
    
    # Filter for markdown files
    md_sermons = [s for s in sermon_list if s['name'].endswith('.md')]
    
    for s_file in md_sermons[:4]: # Take 4 sermons per volume
        s_name = s_file['name']
        download_url = s_file['download_url']
        
        print(f"   📖 Reading {s_name}...")
        content = requests.get(download_url).text
        
        # Split text into paragraphs
        paragraphs = content.split('\n\n')
        vectors_to_upsert = []
        
        for i, para in enumerate(paragraphs):
            clean_text = para.strip()
            # Only index meaningful paragraphs (longer than 150 chars)
            if len(clean_text) > 150:
                vector = model.encode(clean_text).tolist()
                vectors_to_upsert.append({
                    "id": f"{vol_name}_{s_name.replace('.md', '')}_{i}",
                    "values": vector,
                    "metadata": {
                        "text": clean_text[:1000],
                        "source": f"Spurgeon {vol_name} - {s_name}"
                    }
                })
        
        # Push to Pinecone
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            total_sermon_count += 1
        
        # Wait 1 second between sermons to avoid GitHub's secondary rate limits
        time.sleep(1) 

print(f"\n🚀 SUCCESS: {total_sermon_count} sermons across 5 volumes indexed to Pinecone!")
