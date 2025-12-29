import streamlit as st
from transformers import pipeline
import torch

from PyPDF2 import PdfReader


qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
    device=-1
)

def load_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    full_text = ""
    for  page in reader.pages:

        full_text += page.extract_text()
    return full_text

def get_answer(context,question):
    result = qa_pipeline({"context":context,"question":question})
    return result["answer"]
st.set_page_config(page_title="Document QA",layout="wide")
st.title("Intelligent Document Question Answering System")
st.write("Upload a document and ask question directly from it")
uploaded_file = st.file_uploader("Upload a PDF Document",type=["pdf"])
if uploaded_file is not None:
    document_text = load_pdf_text(uploaded_file)
    st.success("Document loaded successfully")
    question = st.text_input("Ask a question fro the document")
    if st.button("Get Answer",use_container_width=True):
        if question.strip() =="":
            st.warning("please enter a question")
        else:
            answer = get_answer(document_text,question)
            st.subheader("Answer")
            st.success(answer)
            


