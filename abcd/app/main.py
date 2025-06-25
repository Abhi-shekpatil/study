import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import os
from app.pipeline import run_pipeline  # To be implemented

# Set up directories
UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="OCR Query Assistant", layout="centered")
st.title("📄 AI-Powered PDF OCR Query Assistant")

# Upload section
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
query = st.text_input("Enter your query for the PDF")

# Save PDF
pdf_filename = None
if uploaded_pdf is not None:
    pdf_filename = uploaded_pdf.name
    save_path = os.path.join(UPLOAD_DIR, pdf_filename)
    with open(save_path, "wb") as f:
        f.write(uploaded_pdf.read())
    st.success(f"Uploaded and saved: {pdf_filename}")

# Run button
if st.button("🔍 Generate Answer"):
    if not uploaded_pdf or not query:
        st.warning("Please upload a PDF and enter a query.")
    else:
        with st.spinner("Processing your query..."):
            response = run_pipeline(pdf_filename, query)
            st.success("Query processed successfully.")
            st.json(response)
