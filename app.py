"""FastAPI application for the Canadian HTS classification workflow."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from graph import run_pipeline, process_escalation_chat

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="HTS Code Generator API")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def get_gemini_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured. Add GEMINI_API_KEY to the .env file.",
        )
    return ChatGoogleGenerativeAI(
        model=os.getenv("HTS_GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=api_key,
        disable_streaming=True,
    )


class ClassifyRequest(BaseModel):
    description: str = Field(min_length=1, max_length=10000)


class EscalationRequest(BaseModel):
    description: str
    search_results: str = ""
    rag_context: str = ""
    chat_history: List[Dict[str, str]] = Field(default_factory=list)


@app.get("/")
def read_index():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))


@app.get("/api/health")
def health():
    return {"status": "ok", "gemini_configured": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))}


@app.post("/api/classify")
def classify_hts(request: ClassifyRequest):
    if not request.description.strip():
        raise HTTPException(status_code=400, detail="Description is required.")
    try:
        result = run_pipeline(request.description.strip(), get_gemini_llm())
        return {
            "success": True,
            "final_hts_code": result.get("final_hts_code"),
            "element_confidences": result.get("element_confidences"),
            "escalation_needed": result.get("escalation_needed"),
            "escalation_question": result.get("escalation_question"),
            "reasoning_steps": result.get("reasoning_steps"),
            "search_results": result.get("search_results"),
            "rag_context": result.get("rag_context"),
            "search_queries": result.get("search_queries"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification failed: {type(exc).__name__}: {exc}") from exc


@app.post("/api/escalation")
def handle_escalation(request: EscalationRequest):
    try:
        result = process_escalation_chat(
            request.description, request.search_results, request.rag_context,
            request.chat_history, get_gemini_llm(),
        )
        return {"success": True, "result": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Escalation failed: {type(exc).__name__}: {exc}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=int(os.getenv("PORT", "8001")), reload=False)
