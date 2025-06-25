from langgraph.graph import StateGraph
from agents.agent_nodes import (
    file_agent_node,
    ocr_agent_node,
    search_agent_node,
    manager_agent_node,
    qa_agent_node,
    save_response_node
)

# Shared state schema (define all possible keys passed between nodes)
state_schema = {
    "pdf_name": str,
    "query": str,
    "session_id": str,
    "file_path": str,
    "image_folder": str,
    "json_path": str,
    "response_path": str,
    "ocr_data": dict,
    "search_result": str,
    "final_response": dict,
    "ocr_method": str,       # NEW: allow selection between "tesseract" or "llava"
    "pdf_id": str            # Used for vectorstore caching
}

def build_graph():
    graph = StateGraph(state_schema)

    # Define agent nodes
    graph.add_node("FileAgent", file_agent_node)
    graph.add_node("OCRAgent", ocr_agent_node)
    graph.add_node("SearchAgent", search_agent_node)
    graph.add_node("ManagerAgent", manager_agent_node)
    graph.add_node("QAAgent", qa_agent_node)
    graph.add_node("SaveResponse", save_response_node)

    # Set graph execution order
    graph.set_entry_point("FileAgent")
    graph.add_edge("FileAgent", "OCRAgent")
    graph.add_edge("OCRAgent", "SearchAgent")
    graph.add_edge("SearchAgent", "ManagerAgent")
    graph.add_edge("ManagerAgent", "QAAgent")
    graph.add_edge("QAAgent", "SaveResponse")
    graph.set_finish_point("SaveResponse")

    return graph.compile()