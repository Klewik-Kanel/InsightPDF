# InsightPDF 📄 | RAG-based Research Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that allows users to chat with PDF documents using Google's Gemini 1.5 Flash and FAISS vector search.

## 🚀 Live Demo
https://insightpdf-l2wejfywbnxdnkzdxgjupy.streamlit.app

## 🛠️ Tech Stack
- **Framework:** Streamlit (UI)
- **Orchestration:** LangChain (LCEL)
- **LLM:** Google Gemini 1.5 Flash
- **Vector Store:** FAISS (Facebook AI Similarity Search)
- **Embeddings:** Google Text-Embedding-004
- **PDF Processing:** PyPDF

## 🧠 How it Works (RAG Pipeline)
1. **Ingestion:** Loads PDF data and splits it into semantic chunks using `RecursiveCharacterTextSplitter`.
2. **Vectorization:** Converts text chunks into high-dimensional numerical vectors.
3. **Storage:** Saves vectors in a FAISS index for lightning-fast similarity search.
4. **Retrieval:** When a user asks a question, the app finds the top 5 most relevant chunks.
5. **Generation:** The LLM reads the context chunks and generates a grounded, hallucination-free answer.

## 🛠️ Local Setup
1. Clone the repo: `git clone https://github.com/Klewik-Kanel/InsightPDF`
2. Install dependencies: `pip install -r requirements.txt`
3. Add your `GOOGLE_API_KEY` to a `.env` file.
4. Run: `streamlit run app.py`
