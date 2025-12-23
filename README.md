# spurgeon-teaching-chat
The goal of this is to make a RAG chat agent that can summarize what Charle's Spurgeon taught in response to particular questions and tell the questioner what sermons they can refer to.

## ⚖️ Scope of Knowledge
The AI's knowledge base at this point is limited to the first five volumes of the *New Park Street Pulpit*.

* **Dataset:** Volumes 1, 2, 3, 4, and 5.
* **Primary Datasource:** https://github.com/lyteword/chspurgeon-sermons/tree/main
* **Sermon Range:** #1 through #285.
* **Historical Window:** 1855 – 1859.
* **Context:** This represents Spurgeon's early ministry in London before the construction of the Metropolitan Tabernacle.

## 🛠️ How It Works
1.  **Semantic Search:** When you ask a question, the system searches a **Pinecone** vector database for the most relevant paragraphs from these specific 285 sermons.
2.  **Grounded Response:** The **Llama-3.3-70b** model (via Groq) processes those snippets and provides a summary.
3.  **Strict Constraints:** The AI is instructed to avoid outside knowledge or general theology. If the answer isn't in these five volumes, the bot will tell you it doesn't know.

## 🧠 Prompting Techniques & AI Logic

This project utilizes several advanced prompting strategies to ensure the assistant remains a faithful research tool rather than a general-purpose AI.

### 1. Retrieval-Augmented Generation (RAG)
Instead of relying on the LLM's internal training (Zero-Shot), we use **Context-Injection**. The system retrieves specific text chunks from Pinecone and feeds them into the prompt. This "grounds" the AI in historical fact.

### 2. Negative Constraint Prompting
To prevent "hallucinations" or the AI adding its own modern theological bias, the system uses strict **Negative Constraints**:
* *Constraint:* "Answer ONLY using the provided context."
* *Constraint:* "Do NOT add outside scriptures or general knowledge."

### 3. Instruction-Tuned Persona
We used **Persona Prompting** to define the bot's voice. While the source material is 19th-century Victorian, the bot is instructed to respond in **Modern English**, making the research accessible without losing the original meaning.

### 4. Grounded Output Formatting
The system is programmed with **Fixed-Output Prompting**. Every response must begin with a specific anchor phrase: *"According to the data I have on his sermons..."* This serves as a constant reminder to the user of the data-limited nature of this exploratory exercise.

## 🛠️ The Tech Stack
* **Language:** Python
* **Framework:** [Streamlit](https://streamlit.io/) (Web Interface)
* **Vector Database:** [Pinecone](https://www.pinecone.io/) (Serverless)
* **LLM:** Llama-3.3-70b via [Groq](https://groq.com/)
* **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers)

---

