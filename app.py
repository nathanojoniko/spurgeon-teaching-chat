import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="Spurgeon Research Bot", page_icon="📚")
st.title("📚 Spurgeon Sermon Archive")
st.subheader("Summary & Reference Tool")

# Initialize APIs
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "spurgeon-teaching-chat"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_KEY)

def search_spurgeon(query):
    query_vector = embed_model.encode(query).tolist()
    # Pulling top 4 matches for a broader summary
    results = index.query(vector=query_vector, top_k=4, include_metadata=True)
    
    context = ""
    sources = []
    for res in results['matches']:
        context += f"\nSOURCE: {res['metadata']['source']}\nCONTENT: {res['metadata']['text']}\n"
        sources.append(res['metadata']['source'])
    
    return context, list(set(sources))

# --- 2. THE CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about a topic (e.g., 'What did he teach about prayer?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        context, sources = search_spurgeon(prompt)
        
        # SYSTEM PROMPT: Focused on summary and objective analysis
        system_instruction = (
            "You are a helpful research assistant specializing in the works of C.H. Spurgeon. "
            "Your goal is to provide a clear, modern summary of his teachings based on the provided excerpts. "
            "1. Do NOT speak in an old-fashioned or poetic style. Use clear, modern English. "
            "2. Summarize the main points across the different sources provided. "
            "3. If the sources mention a specific scripture or analogy, include it. "
            "4. If the answer is not in the context, state that you don't have enough data from the current volumes."
            f"\n\nCONTEXT FROM SERMONS: {context}"
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
        
        # Clear References
        st.markdown("---")
        st.markdown("**Where to find more info:**")
        for s in sources:
            # This creates a clean list of the volumes/files found
            st.info(f"📄 Full Sermon: {s}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
