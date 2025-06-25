import os
from fuzzywuzzy import fuzz
from langchain.schema import HumanMessage

from services.pdf_utils import convert_pdf_to_images
from services.ocr_service import (
    load_or_create_cache,
    build_vectorstore,
    is_vectorstore_cached,
    search_in_vectorstore
)
from services.response_store import save_response
from tools.llm import ask_llm

# 1. File agent: prepares folder
def file_agent_node(state: dict) -> dict:
    os.makedirs(state["image_folder"], exist_ok=True)
    return state

# 2. OCR + Embedding agent
def ocr_agent_node(state: dict) -> dict:
    pdf_name = state["pdf_name"]
    pdf_id = state["pdf_id"]
    file_path = state["file_path"]

    image_folder = os.path.join("data/images", pdf_name)
    json_path = os.path.join("data/ocr_cache", f"{pdf_name}.json")

    image_paths = convert_pdf_to_images(file_path, image_folder)
    ocr_data = load_or_create_cache(image_paths, json_path, state.get("ocr_method", "llava"))

    if not is_vectorstore_cached(pdf_id):
        print(f"[Embedding] Building vectorstore for {pdf_id}")
        build_vectorstore(ocr_data, pdf_id)
    else:
        print(f"[Cache Hit] Vectorstore already exists for {pdf_id}")

    return {
        **state,
        "image_folder": image_folder,
        "ocr_data": ocr_data,
        "json_path": json_path,
    }

# 3. Optional fallback (not wired yet)
def fallback_fuzzy_search(query: str, ocr_data: dict) -> str:
    matches = []
    for page, text in ocr_data.items():
        score = fuzz.partial_ratio(query.lower(), text.lower())
        if score > 50:
            matches.append(f"[{page}] (score: {score})\n{text.strip()[:500]}...")
    return "\n\n".join(matches) if matches else "Not relevant."

# 4. Vector search agent
def search_agent_node(state: dict) -> dict:
    pdf_id = state["pdf_id"]
    query = state["query"]

    context = search_in_vectorstore(pdf_id, query, k=3)
    return {**state, "search_result": context}

# ✅ 5. Query Answering agent using local LLM
from tools.llm import ask_llm  # or wherever you placed it

def qa_agent_node(state: dict) -> dict:
    query = state["query"]
    context = state["search_result"]

    if not context.strip():
        return {
            **state,
            "final_response": {
                "query": query,
                "answer": "No relevant content found to answer the query."
            }
        }

    prompt = (
        f"You are an AI document analyst. Use the OCR-extracted content below to answer the query.\n\n"
        f"Query:\n{query}\n\n"
        f"OCR Content:\n{context}\n\n"
        f"Return the most relevant portion of the content that answers the query, "
    )

    answer = ask_llm(prompt)

    return {
        **state,
        "final_response": {
            "query": query,
            "answer": answer
        }
    }


# 6. Save final response
def save_response_node(state: dict) -> dict:
    save_response(state["response_path"], state["final_response"])
    return state

# 7. Manager agent: orchestrates and logs state
def manager_agent_node(state: dict) -> dict:
    """
    Acts as an orchestrator that logs state metadata and optionally validates OCR step.
    In future, can be extended to reroute or retry failed steps.
    """
    pdf_name = state.get("pdf_name")
    query = state.get("query")
    ocr_data = state.get("ocr_data", {})

    print(f"[Manager Agent] Processing PDF: {pdf_name} | Query: {query}")
    print(f"[Manager Agent] OCR pages found: {len(ocr_data)}")

    if not ocr_data:
        print("[Manager Agent] Warning: No OCR data found.")

    return state
