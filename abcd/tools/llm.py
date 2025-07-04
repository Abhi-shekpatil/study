from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
import requests

# Initialize your local model (already downloaded via Ollama)
llm = ChatOllama(model="llama3.2", temperature=0)

def ask_llm(prompt: str, model: str = "llama3.2") -> str:
    try:
        print(f"[ask_llm] Sending prompt to model '{model}':\n{prompt}\n")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )

        print("[ask_llm] HTTP status:", response.status_code)

        if response.ok:
            try:
                data = response.json()
                print("[ask_llm] Raw response JSON:", data)
                answer = data.get("response", "").strip()
                return answer or "LLM Error: Empty response content."
            except Exception as json_err:
                print("[ask_llm] JSON decoding error:", json_err)
                print("[ask_llm] Raw text response:", response.text)
                return "LLM Error: Failed to parse JSON response."
        else:
            print("[ask_llm] Request failed:", response.text)
            return "LLM Error: Unable to generate response."
    except Exception as e:
        print("[ask_llm] Exception occurred:", e)
        return "LLM Error: Unable to generate response."

