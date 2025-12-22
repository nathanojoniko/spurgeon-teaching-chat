# spurgeon-teaching-chat
The goal of this is to make a RAG chat agent that can summarize what Charle's Spurgeon taught in response to particular questions and tell the questioner what sermons they can refer to.

## ⚖️ Scope of Knowledge
The AI's knowledge base is strictly limited to the first five volumes of the *New Park Street Pulpit*.

* **Dataset:** Volumes 1, 2, 3, 4, and 5.
* **Sermon Range:** #1 through #285.
* **Historical Window:** 1855 – 1859.
* **Context:** This represents Spurgeon's early ministry in London before the construction of the Metropolitan Tabernacle.

## 🛠️ How It Works
1.  **Semantic Search:** When you ask a question, the system searches a **Pinecone** vector database for the most relevant paragraphs from these specific 285 sermons.
2.  **Grounded Response:** The **Llama-3.3-70b** model (via Groq) processes those snippets and provides a summary.
3.  **Strict Constraints:** The AI is instructed to avoid outside knowledge or general theology. If the answer isn't in these five volumes, the bot will tell you it doesn't know.

## 🛠️ The Tech Stack
* **Language:** Python
* **Framework:** [Streamlit](https://streamlit.io/) (Web Interface)
* **Vector Database:** [Pinecone](https://www.pinecone.io/) (Serverless)
* **LLM:** Llama-3.3-70b via [Groq](https://groq.com/)
* **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)

---

