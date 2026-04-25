import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="InsightPDF", page_icon="📄")
st.title("InsightPDF 📄")
st.write("---")

# 1. Sidebar for API Key
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key:", type="password")
    st.info("Get your key at aistudio.google.com")

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    
    # 2. File Upload
    u_file = st.file_uploader("Upload a Research Paper", type="pdf")
    
    if u_file:
        with st.spinner("Analyzing PDF..."):
            with open("temp.pdf", "wb") as f:
                f.write(u_file.getbuffer())
            
            # Load and Split
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            
            # Create Vector Store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vector_index = FAISS.from_documents(pages, embeddings)
            retriever = vector_index.as_retriever()

            # 3. Modern LCEL Chain (The 2026 Way)
            template = """Answer the question based ONLY on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

            # The "Pipe" Chain: Retrieve -> Format -> Prompt -> Model -> Parse
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            st.success("PDF Ready!")
            
            # 4. Chat Interface
            user_query = st.text_input("Ask anything about the document:")
            if user_query:
                with st.spinner("Thinking..."):
                    response = chain.invoke(user_query)
                    st.markdown("### Answer:")
                    st.write(response)
else:
    st.warning("Please enter your API Key in the sidebar to start.")