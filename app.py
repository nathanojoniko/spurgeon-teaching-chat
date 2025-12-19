import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="Spurgeon Research Bot", page_icon="📚")

with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat / Refresh"):
        st.session_state.messages = []
        st.rerun()

# Initialize APIs
PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_KEY = st.secrets["GROQ_API_KEY"]
INDEX_NAME = "spurgeon-teaching-chat"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_KEY)

def search_spurgeon(query):
    query_vector = embed_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=4, include_metadata=True)
    
    context = ""
    source_details = []
    
    for res in results['matches']:
        # Example metadata source format: "volume-1 - sermon-001.md"
        source_id = res['metadata'].get('source', '') 
        raw_text = res['metadata'].get('text', '')
        
        # 1. Construct the Link to the original lyteword repo
        # Format: https://github.com/lyteword/chspurgeon-sermons/blob/main/[path]
        clean_path = source_id.replace("Spurgeon ", "").strip()
        encoded_path = clean_path.replace(" ", "%20")
        github_url = f"https://github.com/lyteword/chspurgeon-sermons/blob/main/{encoded_path}"
        
        # 2. Get a "Pretty Title" 
        # Since we don't have the header in every chunk, we'll clean the filename
        # "volume-1 - sermon-001.md" -> "Sermon 001 (Volume 1)"
        display_name = clean_path.replace(".md", "").replace("-", " ").title()

        context += f"\n--- SOURCE: {display_name} ---\n{raw_text}\n"
        
        source_details.append({
            "title": display_name,
            "url": github_url
        })
    
    return context, source_details

# --- 2. THE CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a topic..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context, source_list = search_spurgeon(prompt)
        
        system_instruction = (
            "You are a research assistant for C.H. Spurgeon's sermons. "
            "STRICT RULES:\n"
            "1. Answer ONLY using the provided context. No outside knowledge.\n"
            "2. Use modern English. No Victorian style.\n"
            "3. Reference the specific Volume or Sermon number within your answer when possible.\n"
            "4. Do NOT provide the final list of links; the app handles that."
            f"\n\nCONTEXT:\n{context}"
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = response.choices[0].message.content
        st.markdown(answer)
        
        # --- REFERENCE SECTION ---
        st.markdown("---")
        st.markdown("### 📖 Original Sermon Files")
        
        # Deduplicate and display
        seen_urls = set()
        for source in source_list:
            if source['url'] not in seen_urls:
                st.markdown(f"🔗 [Read {source['title']}]({source['url']})")
                seen_urls.add(source['url'])

    st.session_state.messages.append({"role": "assistant", "content": answer})
