
import streamlit as st
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.set_page_config(
    page_title="Medical Code Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .title1 {
        color: black;
        text-align: center;
        margin-bottom: 2rem;
    }
    .title {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .input-box {
        margin-bottom: 1rem;
    }
    .answer-box {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #e9ecef;
        border-radius: 10px;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title1'>Medical Code Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='title'>Ask me anything about medical codes!</h2>", unsafe_allow_html=True)

question = st.text_input("Ask a Question", key="input_box", placeholder="Type your question here...")

st.sidebar.header("About")
st.sidebar.info("This Medical Code Chatbot helps you find information about medical codes from provided PDF documents.")

st.sidebar.header("Instructions")
st.sidebar.info(
    """
    1. Upload your PDF files using the uploader below.
    2. Enter your question in the text input box.
    3. The chatbot will generate an answer based on the documents.
    4. The answer will appear below the input box.
    """
)

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

@st.cache_resource
def create_vector_store(uploaded_files):
    '''Create a vector store from uploaded PDF files'''
    # Create temp directory if it does not exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda'})
    db = FAISS.from_documents(texts, embeddings)
    return db

@st.cache_resource
def load_llm():
    llm = LlamaCpp(
        model_path="D:/llama-2-7b-chat.Q4_0.gguf",
        n_gpu_layers=100,
        n_batch=512,
        verbose=False
    )
    return llm

if uploaded_files:
    db = create_vector_store(uploaded_files)
    llm = load_llm()

    template = """
        Context: {context}
        Question: {question}
        Answer:
        """
    prompt = PromptTemplate(template=template, input_variables=['question'])

    retriever = db.as_retriever(search_kwargs={'k': 2})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )

    if question:
        answer = chain({'query': question})['result']
        st.write(question)
        st.text_area("Answer", value=answer, key="input2", placeholder="Answer will appear here", help="The answer to your question will be displayed here.", height=200)
else:
    st.write("Please upload PDF files to proceed.")
