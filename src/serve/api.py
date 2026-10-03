"""Phase 17: local HTTP API for the quantized financial-reasoning model (FastAPI in front of llama-server).

Everything runs on this laptop: llama-server holds the GGUF on the 4 GB GPU, this service builds prompts exactly as in
training (system prompt, report excerpt, question, pinned chat template) and parses the final answer.

Endpoints
  GET  /health    llama-server status
  GET  /model     which GGUF is loaded, quantization, context size
  POST /generate  {"context": "...report text/table...", "question": "..."} -> reasoning steps + parsed answer
  POST /chat      {"messages": [{"role": "user", "content": "..."}]}      -> raw chat completion

Run:
  tools/llama.cpp/b11209/bin/llama-server.exe -m models/orpo_r16/orpo_r16-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080
  uvicorn src.serve.api:app --port 8000
"""

import json
import os
import time
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.render import CHAT_TEMPLATE_KWARGS, SYSTEM_PROMPT
from src.eval.scoring import extract_answer

LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
TOKENIZER = "unsloth/Llama-3.2-3B-Instruct"

app = FastAPI(title="Financial Reasoning SLM", version="1.0")
_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        import src.paths  # noqa: F401
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tokenizer


def _call(path: str, body: dict | None = None, timeout: float = 300) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(LLAMA_URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"llama-server unavailable: {e}")


def _complete(messages: list[dict], max_tokens: int) -> dict:
    tok = tokenizer()
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                     **CHAT_TEMPLATE_KWARGS).removeprefix(tok.bos_token)
    t0 = time.perf_counter()
    out = _call("/completion", {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0, "top_k": 1,
                                "stop": ["<|eot_id|>"], "cache_prompt": True})
    t = out.get("timings", {})
    return {"text": out["content"].strip(),
            "usage": {"prompt_tokens": t.get("prompt_n"), "completion_tokens": t.get("predicted_n")},
            "timing": {"ttft_ms": t.get("prompt_ms"), "gen_tok_s": t.get("predicted_per_second"),
                       "total_ms": round((time.perf_counter() - t0) * 1000, 1)}}


class GenerateRequest(BaseModel):
    context: str = Field(..., description="Report excerpt: text and/or a markdown table")
    question: str
    max_tokens: int = Field(256, ge=1, le=1024)


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = Field(256, ge=1, le=1024)


@app.get("/health")
def health() -> dict:
    return {"api": "ok", "llama_server": _call("/health", timeout=5).get("status")}


@app.get("/model")
def model() -> dict:
    props = _call("/props", timeout=5)
    return {"model_path": props.get("model_path"), "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
            "total_slots": props.get("total_slots"), "build": props.get("build_info")}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{req.context.strip()}\n\nQuestion: {req.question.strip()}"}]
    result = _complete(messages, req.max_tokens)
    result["answer"] = extract_answer(result["text"]) if "answer:" in result["text"].lower() else None
    return result


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    if not req.messages or req.messages[-1].get("role") != "user":
        raise HTTPException(status_code=422, detail="last message must be from the user")
    return _complete(req.messages, req.max_tokens)
