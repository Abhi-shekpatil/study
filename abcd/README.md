# 📄 AI-Powered PDF OCR Query Assistant

This project uses a modular, agent-based pipeline to extract text from uploaded PDFs and answer queries about their contents using OCR and LangGraph-powered AI agents.

---

## 🚀 Features

- Upload PDFs via Streamlit
- Convert PDF pages to images
- Run OCR using Tesseracts
- Store extracted text using Pickle
- Query the text with LangChain + LangGraph agents
- Validate responses with a QA agent
- Save results as JSON and display in the UI

---

## 🧠 Agent Graph (LangGraph Flow)

graph TD
    Start --> FileAgent
    FileAgent --> OCRAgent
    OCRAgent --> SearchAgent
    SearchAgent --> ManagerAgent
    ManagerAgent --> QAAgent
    QAAgent --> SaveResponse
    SaveResponse --> End 

---

## 🗂️ Project Structure

ocr_ai_pipeline/
├── app/
│   ├── main.py              # Streamlit UI
│   └── pipeline.py          # Runs LangGraph pipeline
│
├── agents/
│   ├── __init__.py
│   ├── tools.py             # LangChain tools
│   ├── agent_nodes.py       # Node logic
│   └── langgraph_config.py  # Graph definition
│
├── services/
│   ├── pdf_utils.py
│   ├── ocr_service.py
│   └── response_store.py
│
├── data/
│   ├── uploads/             # Uploaded PDFs
│   ├── images/              # Extracted page images
│   ├── ocr_cache/           # Pickled OCR results
│   └── responses/           # Final answers
│
├── requirements.txt
└── README.md

---

## 🧪 Usage

### Install dependencies
pip install -r requirements.txt

### Ensure Poppler is installed
macOS: brew install poppler
Ubuntu/Debian: sudo apt install poppler-utils

### Run the app
streamlit run app/main.py

### Upload a PDF, enter a query, and hit "Generate Answer".

---

## 🧰 Powered By

LangGraph
LangChain
Streamlit
Tesseract OCR