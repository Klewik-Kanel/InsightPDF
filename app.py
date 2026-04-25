import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="InsightPDF")
st.title("InsightPDF ")

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ API Key not found in Secrets!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 2. AUTO-DISCOVERY ENGINE ---
@st.cache_resource
def get_models():
    """Finds the best available models supported by this API key."""
    try:
        all_models = list(genai.list_models())
        
        # Find Embedding Model
        embed_list = [m.name for m in all_models if 'embedContent' in m.supported_generation_methods]
        final_embed = "models/text-embedding-004" if "models/text-embedding-004" in embed_list else embed_list[0]
        
        # Find Chat Model (Looking for Flash 1.5 variants)
        chat_list = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
        # Prefer Flash 1.5, then Pro, then whatever is first
        final_chat = next((m for m in chat_list if "gemini-1.5-flash" in m), chat_list[0])
        
        return final_embed, final_chat
    except Exception as e:
        st.error(f"Discovery failed: {e}")
        return "models/embedding-001", "models/gemini-pro"

embed_name, chat_name = get_models()

# --- 3. UI & RAG LOGIC ---
u_file = st.file_uploader("Upload PDF", type="pdf")

if u_file:
    with st.spinner(f"Using {embed_name} to index..."):
        with open("temp.pdf", "wb") as f:
            f.write(u_file.getbuffer())
        
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        embeddings = GoogleGenerativeAIEmbeddings(model=embed_name)
        vector_index = FAISS.from_documents(pages, embeddings)
        retriever = vector_index.as_retriever(search_kwargs={"k": 5})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        template = """Answer strictly based on context. 
        Context: {context}
        Question: {question}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize the discovered Chat Model
        llm = ChatGoogleGenerativeAI(
            model=chat_name,
            temperature=0.1,
            safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        )

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        
    st.success(f"Ready! (Powered by {chat_name})")
    
    user_query = st.chat_input("Ask about the PDF...")
    if user_query:
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            response = chain.invoke(user_query)
            st.write(response)