import streamlit as st
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# --- 1. SETUP & INITIALIZATION ---
st.set_page_config(page_title="Spurgeon Research Bot", page_icon="📚")

# Sidebar for controls
with st.sidebar:
    st.title("Settings")
    st.info("This bot searches the first 5 volumes of Spurgeon's sermons to provide factual summaries and references.")
    if st.button("Clear Chat / Refresh"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Spurgeon Sermon Archive")
st.subheader("Summary & Reference Tool")

# Access API keys from Streamlit Secrets
PINECONE_KEY = st.secrets["PINECONE_API_KEY"]
GROQ_KEY = st.secrets["GROQ_API_KEY"]
INDEX_NAME = "spurgeon-teaching-chat"

# Initialize Clients
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
groq_client = Groq(api_key=GROQ_KEY)

def search_spurgeon(query):
    """Searches Pinecone and formats the context and GitHub URLs."""
    query_vector = embed_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=4, include_metadata=True)
    
    context = ""
    source_details = []
    
    for res in results['matches']:
        # metadata format from your upload: "Spurgeon volume-X - sermon-XXX.md"
        source_id = res['metadata'].get('source', '') 
        raw_text = res['metadata'].get('text', '')
        
        # --- PATH CORRECTION LOGIC ---
        # 1. Clean the prefix
        clean_path = source_id.replace("Spurgeon ", "").strip()
        
        # 2. Fix GitHub structure: volume-x/sermon-xxx.md
        # We replace the FIRST occurrence of " - " with "/"
        if " - " in clean_path:
            correct_github_path = clean_path.replace(" - ", "/", 1)
        else:
            correct_github_path = clean_path
            
        # 3. Handle URL encoding (spaces to %20)
        encoded_path = correct_github_path.replace(" ", "%20")
        github_url = f"https://github.com/lyteword/chspurgeon-sermons/blob/main/{encoded_path}"
        
        # 4. Create Readable Title (e.g., "Volume 1 Sermon 001")
        display_name = clean_path.replace(".md", "").replace("-", " ").title()

        context += f"\n--- SOURCE: {display_name} ---\n{raw_text}\n"
        
        source_details.append({
            "title": display_name,
            "url": github_url
        })
    
    return context, source_details

# --- 2. CHAT HISTORY HANDLING ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. CHAT INTERACTION ---
if prompt := st.chat_input("Ask a question about Spurgeon's teaching..."):
    # Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieve context from Pinecone
        context, source_list = search_spurgeon(prompt)
        
        # STRICT SYSTEM PROMPT
        system_instruction = (
            "You are a research assistant for the sermons of C.H. Spurgeon. "
            "STRICT RULES:\n"
            "1. Answer ONLY using the provided context. Do NOT use outside knowledge.\n"
            "2. Use clear, modern English. Do NOT mimic Victorian style.\n"
            "3. Start your response EXACTLY with the phrase: 'Based on the data I have on Spurgeon's sermons...'\n"
            "4. Do NOT cite volume or sermon numbers inside your summary text.\n"
            "5. Do NOT provide the final list of links or a bibliography; the app handles that."
            "6. Use bullet points where necessary to make text readable."
            "7. REFUSAL RULE: If the answer is not in the context, your ENTIRE response must ONLY be: "
            "'Based on the data I have on Spurgeon's sermons, there is no information available regarding [TOPIC].'\n"
            "8. ABSOLUTELY NO FILLER: Do not explain what the context *does* contain or why the information is missing."
            f"\n\nCONTEXT FROM SERMONS:\n{context}"
        )

        # Generate LLM response
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
        if source_list:
            st.markdown("---")
            st.write("Below are the sermons I referenced in giving this summary. Please read them to get more information on his teaching regarding this subject:")
            
            # Deduplicate links in case multiple chunks came from the same sermon
            seen_urls = set()
            for source in source_list:
                if source['url'] not in seen_urls:
                    st.markdown(f"🔗 [Read {source['title']}]({source['url']})")
                    seen_urls.add(source['url'])

    # Save assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": answer})
