import os
import json
import re
import unicodedata
import base64
import pickle
import pytesseract
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor, as_completed


# Optional for llava-based OCR
import requests

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------------
# TEXT UTILS
# ----------------------------
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"\n(?!\n)", " ", text)  # flatten single line breaks
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()'\"]+", " ", text)  # remove weird chars
    return text.strip()

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ----------------------------
# OCR ENGINE
# ----------------------------
def extract_text_tesseract(image_paths: list[str]) -> dict:
    results = {}
    for img_path in image_paths:
        text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
        results[os.path.basename(img_path)] = clean_text(text)
    return results

def extract_text_llava(image_paths: list[str], max_workers: int = 8) -> dict:
    """
    Parallel OCR using LLaVA via Ollama API with fallback to Tesseract for failed pages.
    """
    def process_image(img_path):
        try:
            img_base64 = encode_image_to_base64(img_path)
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llava",
                    "prompt": "Extract all the readable text from this image in clean form.",
                    "images": [img_base64]
                },
                timeout=30
            )

            if response.ok:
                text = clean_text(response.json().get("response", ""))
                if not text.strip():
                    raise ValueError("Empty response from LLaVA")
                return os.path.basename(img_path), text
            else:
                raise RuntimeError(f"Failed HTTP: {response.status_code}")

        except Exception as e:
            print(f"[Fallback:Tesseract] {img_path} -> {e}")
            fallback_text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
            return os.path.basename(img_path), clean_text(fallback_text)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(process_image, path): path for path in image_paths}
        for future in as_completed(future_to_path):
            fname, content = future.result()
            results[fname] = content

    return results


# ----------------------------
# CACHE + OCR INTERFACE
# ----------------------------
def load_or_create_cache(image_paths: list, json_path: str, method: str = "llava") -> dict:
    """
    Loads OCR results from JSON if available; otherwise runs OCR (LLaVA or Tesseract) and saves.
    Includes fallback to Tesseract if LLaVA fails.
    """
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)

    text_data = {}

    try:
        if method == "llava":
            print("[OCR] Using LLaVA for OCR")
            results = extract_text_llava(image_paths)

            # Check if LLaVA returned meaningful text
            empty_pages = [k for k, v in results.items() if not v.strip() or "[Error" in v]
            if len(empty_pages) == len(image_paths):
                raise RuntimeError("LLaVA OCR failed on all pages. Falling back to Tesseract.")

            text_data = results
        else:
            raise ValueError("Skip to Tesseract")

    except Exception as e:
        print(f"[Fallback] OCR failed with LLaVA. Reason: {e}")
        print("[OCR] Falling back to Tesseract...")

        for img_path in image_paths:
            text = pytesseract.image_to_string(Image.open(img_path), lang="eng")
            text_data[os.path.basename(img_path)] = text

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(text_data, f, indent=2)

    return text_data


# ----------------------------
# VECTORSTORE
# ----------------------------
def build_vectorstore(text_dict: dict, pdf_id: str):
    combined_text = "\n".join(text_dict.values())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([combined_text])

    vectorstore = FAISS.from_documents(docs, embedding_model)

    folder = f"data/vectorstore/{pdf_id}"
    os.makedirs(folder, exist_ok=True)
    vectorstore.save_local(folder)

    with open(os.path.join(folder, "docs.pkl"), "wb") as f:
        pickle.dump(docs, f)

def search_in_vectorstore(pdf_id: str, query: str, k: int = 3) -> str:
    folder = f"data/vectorstore/{pdf_id}"
    vectorstore = FAISS.load_local(folder, embedding_model, allow_dangerous_deserialization=True)

    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])

def is_vectorstore_cached(pdf_id: str) -> bool:
    folder = f"data/vectorstore/{pdf_id}"
    return os.path.exists(os.path.join(folder, "index.faiss")) and os.path.exists(os.path.join(folder, "docs.pkl"))
