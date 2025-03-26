# streamlit_app.py
import streamlit as st
import requests

BASE_URL = "http://localhost:8000"

st.title("RAG CSV Analyser")

# Upload
uploaded_file = st.file_uploader("Upload CSV")
if uploaded_file:
    response = requests.post(f"{BASE_URL}/upload", files={"file": uploaded_file})
    if response.status_code == 200:
        file_id = response.json()["file_id"]
        st.success(f"File uploaded! ID: {file_id}")

# List files
if st.button("List Files"):
    response = requests.get(f"{BASE_URL}/files")
    files = response.json()["files"]
    st.write(files)

# Query
file_id = st.text_input("File ID")
query = st.text_input("Query")
if st.button("Ask"):
    response = requests.post(f"{BASE_URL}/query", json={"file_id": file_id, "query": query})
    st.write(response.json()["response"])