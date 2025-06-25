import os
from agents.langgraph_config import build_graph
from uuid import uuid4

def run_pipeline(pdf_filename: str, query: str, ocr_method: str = "llava") -> dict:
    """
    Orchestrates the full pipeline using LangGraph and agent nodes.
    `ocr_method`: either "tesseract" (for faster, local OCR) or "llava" (for richer, AI-based OCR)
    """

    # Generate unique run/session ID for this query
    session_id = str(uuid4())[:4]
    pdf_id = os.path.splitext(pdf_filename)[0]

    # Prepare initial pipeline state
    state = {
        "pdf_name": pdf_filename,
        "pdf_id": pdf_id,
        "query": query,
        "session_id": session_id,
        "ocr_method": ocr_method,  # 👈 NEW: pass OCR method
        "file_path": os.path.join("data/uploads", pdf_filename),
        "image_folder": os.path.join("data/images", pdf_id),
        "json_path": os.path.join("data/ocr_cache", f"{pdf_id}.json"),
        "response_path": os.path.join("data/responses", f"{pdf_id}_response.json")
    }

    # Build and run LangGraph workflow
    graph = build_graph()
    final_state = graph.invoke(state)

    return final_state.get("final_response", {
        "status": "error",
        "message": "No response returned from agent pipeline."
    })
