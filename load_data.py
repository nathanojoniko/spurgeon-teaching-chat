import os
import requests
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# 1. Setup our "Librarian" tools
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat" # Your exact name!

# Connect to the Library
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)

# This model turns text into 384 numbers (matching your Index!)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Pick a sermon to read (The Immutability of God - 1855)
# This link goes straight to a plain-text version of his first published sermon
url = "https://www.spurgeongems.org/wp-content/uploads/2013/05/chs1.txt"
print(f"Reading sermon from: {url}")
response = requests.get(url)
text = response.text

# 3. Chop the sermon into bite-sized paragraphs (Chunks)
# AI works better when it reads one paragraph at a time
paragraphs = text.split('\n\n') 

print(f"I found {len(paragraphs)} paragraphs. Turning them into math vectors...")

# 4. Save them to the library
for i, para in enumerate(paragraphs):
    if len(para.strip()) > 50: # Only save real sentences, skip empty lines
        vector = model.encode(para).tolist()
        # We give each chunk a unique ID and save the text as "metadata"
        index.upsert(vectors=[{
            "id": f"sermon1_para{i}", 
            "values": vector, 
            "metadata": {"text": para, "source": "Sermon #1"}
        }])

print("Successfully loaded your first sermon into the library!")
