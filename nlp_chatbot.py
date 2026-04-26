import streamlit as st
import os
import fitz 
import re
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# Page Config
st.set_page_config(page_title="Finance AI Chatbot", page_icon="💰")
st.title("📑 Finance RAG Chatbot")
st.markdown("Ask questions based on Basel III, Credit Risk, and more.")

# 1. LOAD MODELS (Cached to prevent reloading on every click)
@st.cache_resource
def load_resources():
    nlp = spacy.load("en_core_web_sm")
    
    # Load your fine-tuned model
    # Note: Ensure this folder is in your GitHub repo
    model_path = "./fine_tuned_flan_t5_finance" 
    if not os.path.exists(model_path):
        model_path = "google/flan-t5-base" # Fallback
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return nlp, tokenizer, model

nlp, t5_tokenizer, t5_model = load_resources()

# 2. DATA LOADING & INDEXING
@st.cache_data
def process_pdfs():
    pdf_folder = "./data" # Place your PDFs here in GitHub
    documents = []
    if os.path.exists(pdf_folder):
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                doc = fitz.open(os.path.join(pdf_folder, filename))
                text = "".join([page.get_text() for page in doc])
                documents.append(text)
    
    # Simple Chunking
    all_chunks = []
    for doc_text in documents:
        words = doc_text.split()
        chunks = [" ".join(words[i:i + 400]) for i in range(0, len(words), 400)]
        all_chunks.extend(chunks)
    
    # Vectorizing
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    return all_chunks, vectorizer, tfidf_matrix

chunks, vectorizer, tfidf_matrix = process_pdfs()

# 3. CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("What is the CET1 ratio under Basel III?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieval
    q_vec = vectorizer.transform([prompt.lower()])
    sim = cosine_similarity(q_vec, tfidf_matrix)
    best_idx = sim.argmax()
    context = chunks[best_idx]

    # Generation
    input_text = f"context: {context} question: {prompt}"
    inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = t5_model.generate(**inputs, max_new_tokens=100)
    response = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Display Assistant Response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})