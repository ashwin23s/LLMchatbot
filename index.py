import streamlit as st
import base64

st.title("Medical Code Chatbot")
st.text_input("Question", key="input1", placeholder="Type your question here", help="Enter your medical question here.")
st.text_input("Answer", key="input2", placeholder="Answer will appear here", help="The answer to your question will be displayed here.")
