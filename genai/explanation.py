import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# PROJECT ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from genai.prompts import DETECTION_EXPLANATION_PROMPT

API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def get_genai_chain():
    """
    Lazy loader for LangChain + Gemini LLM.
    Returns chain or None if API key is not configured.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key.strip() == "":
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
            temperature=0.2
        )
        prompt = ChatPromptTemplate.from_template(DETECTION_EXPLANATION_PROMPT)
        return prompt | llm
    except Exception as e:
        print(f"GenAI setup error: {e}")
        return None

def generate_explanation(text, detection_result):
    """
    Generate an AI analysis report using Gemini.
    """
    chain = get_genai_chain()
    if not chain:
        return (
            "⚠️ **GenAI Explanation Unavailable**\n\n"
            "To enable Gemini-powered detailed explanations, please ensure `GOOGLE_API_KEY` "
            "is properly configured in your `.env` file."
        )

    try:
        response = chain.invoke({
            "text": text[:1500],  # Trim long text to speed up response
            "prediction": detection_result.get("prediction", "Unknown"),
            "ai_probability": detection_result.get("ai_probability", 0),
            "human_probability": detection_result.get("human_probability", 0),
            "linguistic_features": detection_result.get("linguistic_features", {}),
            "burstiness": detection_result.get("burstiness", {}),
            "perplexity": detection_result.get("perplexity", 0)
        })

        content = response.content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(text_parts).strip()

        return str(content).strip()
    except Exception as e:
        return f"⚠️ **GenAI Explanation Error**: {str(e)}"

if __name__ == "__main__":
    sample = {"prediction": "Human Written", "ai_probability": 10.0, "human_probability": 90.0, "linguistic_features": {}, "burstiness": {}, "perplexity": 45.2}
    print(generate_explanation("Sample text input.", sample))