import streamlit as st
import os
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. CONFIGURATION & API KEY ---
st.set_page_config(page_title="InsightPDF", page_icon="📄")
st.title("InsightPDF 📄")

# Look for API key in Streamlit Secrets (Cloud) or Environment (Local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("⚠️ API Key not found! Please add it to Streamlit Secrets or a .env file.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 2. DYNAMIC MODEL SELECTION ---
# This avoids the 404 error by finding what your key actually supports
@st.cache_resource
def get_embedding_model():
    try:
        available_models = [m.name for m in genai.list_models() if 'embedContent' in m.supported_generation_methods]
        # Prefer text-embedding-004, fallback to first available
        chosen = "models/text-embedding-004" if "models/text-embedding-004" in available_models else available_models[0]
        return GoogleGenerativeAIEmbeddings(model=chosen)
    except Exception as e:
        # Hard fallback if list fails
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()

# --- 3. UI & RAG LOGIC ---
u_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if u_file:
    with st.spinner("Processing document..."):
        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(u_file.getbuffer())
        
        # Load and Index
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        vector_index = FAISS.from_documents(pages, embeddings)
        retriever = vector_index.as_retriever(search_kwargs={"k": 5})

        # Define the Chain
        template = """Answer the question based strictly on the context provided. 
        If the answer isn't in the context, say you don't know.
        
        Context: {context}
        Question: {question}"""
        
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        
    st.success("Analysis Complete!")
    
    # Chat Input
    user_query = st.chat_input("What would you like to know?")
    if user_query:
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            response = chain.invoke(user_query)
            st.write(response)