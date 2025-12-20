import os
import requests
import time
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# --- 1. SETUP & AUTHENTICATION ---
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
# Create a secret/env variable for this on your machine or GitHub Actions
GITHUB_TOKEN = os.getenv("GH_ACCESS_TOKEN") 

INDEX_NAME = "spurgeon-teaching-chat"
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Headers for GitHub API to avoid rate limiting
headers = {"Authorization": f"token {GH_ACCESS_TOKEN}"} if GH_ACCESS_TOKEN else {}

REPO_API_URL = "https://api.github.com/repos/lyteword/chspurgeon-sermons/contents/"

# --- 2. FETCH VOLUMES ---
print("📥 Fetching full Volume list...")
response = requests.get(REPO_API_URL, headers=headers)
repo_contents = response.json()

if isinstance(repo_contents, dict) and "message" in repo_contents:
    print(f"❌ GitHub API Error: {repo_contents['message']}")
    exit()

volume_folders = [item for item in repo_contents if item['type'] == 'dir' and 'volume-' in item['name']]
print(f"✅ Found {len(volume_folders)} volumes. Beginning FULL index...")

total_vectors_upserted = 0

# --- 3. PROCESS ALL VOLUMES ---
for vol in volume_folders:
    vol_name = vol['name']
    vol_url = vol['url']
    
    print(f"\n📂 Processing {vol_name}...")
    vol_response = requests.get(vol_url, headers=headers)
    sermon_list = vol_response.json()
    
    md_sermons = [s for s in sermon_list if s['name'].endswith('.md')]
    
    for s_file in md_sermons:
        s_name = s_file['name']
        download_url = s_file['download_url']
        
        print(f"  📖 {s_name}", end=" ", flush=True)
        try:
            # Use headers here too if the repo is private, otherwise optional but good practice
            content = requests.get(download_url, headers=headers).text
        except Exception as e:
            print(f"❌ Error downloading: {e}")
            continue
            
        paragraphs = content.split('\n\n')
        batch = []
        
        for i, para in enumerate(paragraphs):
            clean_text = para.strip()
            if len(clean_text) > 150:
                vector = model.encode(clean_text).tolist()
                batch.append({
                    "id": f"{vol_name}_{s_name.replace('.md', '')}_{i}",
                    "values": vector,
                    "metadata": {
                        "text": clean_text[:3000],
                        "source": f"Spurgeon {vol_name} - {s_name}"
                    }
                })
                
                # Upsert in batches of 100
                if len(batch) >= 100:
                    index.upsert(vectors=batch)
                    total_vectors_upserted += len(batch)
                    batch = []
        
        if batch:
            index.upsert(vectors=batch)
            total_vectors_upserted += len(batch)
            
        print(f"✅ ({len(paragraphs)} paras)")
        time.sleep(0.2) # Faster sleep with token

print(f"\n🚀 FINISHED! Total vectors in Pinecone: {total_vectors_upserted}")
