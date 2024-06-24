import streamlit as st
import os
# Install required packages
os.system('pip install pypdf')
os.system('pip install langchain')
os.system('pip install sentence-transformers')
os.system('pip install faiss-cpu')
os.system('pip install langchain-community')
os.system('pip install llama-cpp-python')

from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up the document loader and load documents
loader = DirectoryLoader(path="downloads", glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Set up embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Create a FAISS vector store
db = FAISS.from_documents(texts, embeddings)

# Set up the language model
llm = LlamaCpp(
    model_path="downloads/llama-2-7b-chat.Q4_0.gguf",
    n_gpu_layers=40,
    n_batch=512,
    verbose=False
)

# Set up the prompt template
template = """
Context: {context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

# Set up the retriever and chain
retriever = db.as_retriever(search_kwargs={'k': 2})
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={'prompt': prompt}
)

# Streamlit UI
st.title('Document QA System')

question = st.text_input("Enter your question:")
if question:
    answer = chain({'query': question})['result']
    st.write('Answer:', answer)
