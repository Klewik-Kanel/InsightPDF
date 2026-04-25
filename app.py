import streamlit as st

# ADD THIS AT THE VERY TOP
st.write("### 🚀 InsightPDF Status: Online") 
st.write("If you see this, the app is running. Waiting for API Key...")

# ... (rest of your code)


import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

st.title("InsightPDF")

# 1. Input for API Key
api_key = st.text_input("Paste your Gemini API Key here:", type="password")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 2. File Upload
    u_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if u_file:
        # Save file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(u_file.getbuffer())
        
        # 3. Process PDF (The RAG part)
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load_and_split()
        
        # Turn text into numbers
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_index = FAISS.from_documents(pages, embeddings)
        
        # 4. Ask a Question
        query = st.text_input("What do you want to know from this PDF?")
        
        if query:
            # Find relevant pages
            docs = vector_index.similarity_search(query)
            
            # Ask the AI
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            chain = load_qa_chain(model, chain_type="stuff")
            response = chain.invoke({"input_documents": docs, "question": query})
            
            st.write(response["output_text"])   