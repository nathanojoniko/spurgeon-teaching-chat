import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="Spurgeon Research Bot", page_icon="📚")

# Sidebar with Refresh Button
with st.sidebar:
    st.title("Settings")
    if st.button("Clear Chat / Refresh"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Spurgeon Sermon Archive")
st.subheader("Summary & Reference Tool")

# Initialize APIs (Ensure these match your Secrets names)
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
    sources = []
    for res in results['matches']:
        # Extract metadata
        src_name = res['metadata'].get('source', 'Unknown Sermon')
        txt_body = res['metadata'].get('text', '')
        context += f"\n--- DOCUMENT: {src_name} ---\n{txt_body}\n"
        sources.append(src_name)
    
    return context, list(set(sources))

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
        context, sources = search_spurgeon(prompt)
        
        # STRICT SYSTEM PROMPT
        system_instruction = (
            "You are a helpful research assistant for the sermons of C.H. Spurgeon. "
            "Your goal is to provide a modern summary based ONLY on the provided documents. "
            "STRICT RULES:\n"
            "1. Use clear, modern English. Do NOT mimic a Victorian style.\n"
            "2. ONLY use the 'CONTEXT FROM SERMONS' provided below. If it is not there, say you don't know.\n"
            "3. DO NOT add your own scriptures or general knowledge (e.g., do not add Philippians 4:6 unless it is in the text).\n"
            "4. At the very end of your response, list the 'Sermons Referenced' clearly.\n"
            f"\n\nCONTEXT FROM SERMONS:\n{context}"
        )

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ]
        )
        
        full_response = response.choices[0].message.content
        st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
