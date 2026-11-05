"""Windows CPU chat backend for Sxarf AI."""

import argparse
import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

MODEL_ID = "llama32"
MODEL_NAME = "Llama 3.2 3B Instruct (GGUF Q4_K_M, CPU)"
MODEL_REPO = "bartowski/Llama-3.2-3B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
MAX_CONTEXT_TOKENS = 4096
MAX_GENERATION_TOKENS = 4096

model = None
model_status = "starting"
model_error = None
generation_lock = threading.Lock()


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = Field(default=1024, ge=1, le=MAX_GENERATION_TOKENS)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    thinking: bool = False
    agents: str = "off"


class ModelRequest(BaseModel):
    model: str


def load_model():
    global model, model_status, model_error

    model_status = "loading"
    model_error = None
    try:
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            cache_dir=os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            ),
        )
        model = Llama(
            model_path=model_path,
            n_ctx=MAX_CONTEXT_TOKENS,
            n_batch=256,
            n_threads=max(1, (os.cpu_count() or 2) - 2),
            n_gpu_layers=0,
            use_mmap=True,
            verbose=False,
        )
        model_status = "ready"
        print(f"[CPU] Ready: {MODEL_NAME}", flush=True)
    except Exception as exc:
        model_error = str(exc)
        model_status = "error"
        raise


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def load_in_background():
        try:
            load_model()
        except Exception as exc:
            print(f"[CPU] Model startup failed: {exc}", flush=True)

    threading.Thread(
        target=load_in_background,
        name="model-loader",
        daemon=True,
    ).start()
    yield


app = FastAPI(title="Sxarf AI Windows CPU Backend", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health():
    return {
        "status": model_status,
        "model": MODEL_NAME if model is not None else None,
        "model_id": MODEL_ID if model is not None else None,
        "repo": MODEL_REPO,
        "error": model_error,
        "context_limit": MAX_CONTEXT_TOKENS,
    }


@app.get("/models")
def get_models():
    return {
        "models": [{"id": MODEL_ID, "name": MODEL_NAME, "repo": MODEL_REPO}],
        "active": MODEL_ID if model is not None else None,
        "status": model_status,
    }


@app.post("/select-model")
def select_model(request: ModelRequest):
    if request.model not in {MODEL_ID, "qwen3"}:
        raise HTTPException(
            status_code=400,
            detail=f"{request.model!r} is not available in the Windows build.",
        )
    if model is None:
        raise HTTPException(status_code=503, detail="The local model is not ready.")
    return {"success": True, "active": MODEL_ID, "name": MODEL_NAME}


def _generate(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="The local model is not ready.")

    messages = [message.model_dump() for message in request.messages]
    with generation_lock:
        try:
            prompt_tokens = len(
                model.tokenize(
                    "\n".join(
                        f"{message['role']}: {message['content']}"
                        for message in messages
                    ).encode("utf-8"),
                    add_bos=True,
                )
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not estimate the conversation size: {exc}",
            ) from exc
    available_tokens = (
        MAX_CONTEXT_TOKENS - prompt_tokens - max(16, len(messages) * 8)
    )
    if available_tokens < 1:
        raise HTTPException(
            status_code=400,
            detail="This conversation is too long for the Windows model context. Start a new chat or shorten the prompt.",
        )
    max_tokens = min(request.max_tokens, available_tokens)

    started = time.perf_counter()
    generated = []
    first_token_sec = 0.0
    output = ""
    output_tokens = 0
    with generation_lock:
        try:
            chunks = model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True,
            )
            for chunk in chunks:
                delta = chunk["choices"][0].get("delta", {}).get("content")
                if delta:
                    if not generated:
                        first_token_sec = time.perf_counter() - started
                    generated.append(delta)
                    yield {"delta": delta}
            output = "".join(generated)
            output_tokens = len(
                model.tokenize(output.encode("utf-8"), add_bos=False)
            )
        except Exception as exc:
            yield {"error": f"Generation failed: {exc}", "status": 500}
            return

    elapsed = time.perf_counter() - started
    yield {
        "done": True,
        "content": output,
        "tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "time_sec": round(elapsed, 2),
        "first_token_sec": round(first_token_sec, 3),
        "tokens_per_sec": round(output_tokens / elapsed, 1) if elapsed else 0,
        "model": MODEL_NAME,
        "context_limit": MAX_CONTEXT_TOKENS,
        "note": "Windows build: CPU inference; research agents are unavailable.",
    }


def _sse(payload):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_response(request: ChatRequest):
    def events():
        for event in _generate(request):
            yield _sse(event)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    return _stream_response(request)


@app.post("/chat/agent/stream")
def agent_chat_stream():
    raise HTTPException(
        status_code=501,
        detail="Research agents are not available in the Windows build.",
    )


@app.post("/chat")
def chat(request: ChatRequest):
    content = ""
    metadata = {}
    for event in _generate(request):
        if "error" in event:
            raise HTTPException(status_code=500, detail=event["error"])
        if "delta" in event:
            content += event["delta"]
        if event.get("done"):
            metadata = event
    return {**metadata, "content": content}


@app.post("/vision/analyze")
def vision_analyze():
    raise HTTPException(
        status_code=501,
        detail="Image analysis is not available in the Windows build.",
    )


@app.post("/speech/transcribe")
def transcribe_speech():
    raise HTTPException(
        status_code=501,
        detail="Local speech transcription is not available in the Windows build.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8741)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
