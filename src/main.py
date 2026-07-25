import time
import asyncio
from contextlib import asynccontextmanager

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils import config, setup_logger, get_model, get_index, get_questions, reset_caches, clean_text

logger = setup_logger(__name__)


# --------------------------------------------------------------------------
# Lifespan: fail fast at startup if artifacts are missing, instead of
# 500-ing on the first request.
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up model/index/corpus...")
    try:
        get_model()
        get_index()
        get_questions()
    except FileNotFoundError as e:
        logger.error(f"Startup failed -- missing artifact: {e}")
        raise
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Duplicate Question Detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before real production exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K)


class SearchResult(BaseModel):
    question: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    latency_ms: float


class PredictRequest(BaseModel):
    question1: str = Field(..., min_length=1, max_length=1000)
    question2: str = Field(..., min_length=1, max_length=1000)


class PredictResponse(BaseModel):
    question1: str
    question2: str
    similarity: float
    is_duplicate: bool
    threshold: float
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    device: str
    indexed_questions: int


class LiveResult(BaseModel):
    title: str
    link: str
    snippet: str


class LiveSearchResponse(BaseModel):
    query: str
    results: list[LiveResult]
    source: str = "google_custom_search"
    latency_ms: float


class CombinedSearchResponse(BaseModel):
    query: str
    local_results: list[SearchResult]
    live_results: list[LiveResult]
    live_search_error: str | None = None
    latency_ms: float


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        index = get_index()
        return HealthResponse(
            status="ok",
            device=config.DEVICE,
            indexed_questions=index.ntotal,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.time()

    model = get_model()
    index = get_index()
    questions = get_questions()

    query = clean_text(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning.")

    top_k = min(req.top_k, index.ntotal)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=config.DEVICE,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, top_k)

    results = [
        SearchResult(question=questions.iloc[idx]["question"], similarity=float(score))
        for score, idx in zip(scores[0], indices[0])
        if idx != -1
    ]

    return SearchResponse(
        query=req.query,
        results=results,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()

    model = get_model()

    q1 = clean_text(req.question1)
    q2 = clean_text(req.question2)
    if not q1 or not q2:
        raise HTTPException(status_code=400, detail="Both questions must be non-empty.")

    embeddings = model.encode(
        [q1, q2],
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=config.DEVICE,
    ).astype(np.float32)

    # Normalized embeddings -> dot product == cosine similarity.
    similarity = float(np.dot(embeddings[0], embeddings[1]))

    return PredictResponse(
        question1=req.question1,
        question2=req.question2,
        similarity=similarity,
        is_duplicate=similarity >= config.DUPLICATE_THRESHOLD,
        threshold=config.DUPLICATE_THRESHOLD,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


async def _google_search_quora(query: str) -> list[LiveResult]:
    """Query Google Custom Search restricted to quora.com.

    Raises RuntimeError with a human-readable message on any failure
    (missing config, network error, quota exceeded) so callers can decide
    whether to fail hard (/live-search) or degrade gracefully (/combined-search).
    """
    if not config.GOOGLE_API_KEY or not config.GOOGLE_SEARCH_ENGINE_ID:
        raise RuntimeError(
            "Google Custom Search is not configured. Set GOOGLE_API_KEY and "
            "GOOGLE_SEARCH_ENGINE_ID environment variables."
        )

    params = {
        "key": config.GOOGLE_API_KEY,
        "cx": config.GOOGLE_SEARCH_ENGINE_ID,
        "q": f"site:quora.com {query}",
        "num": 5,
    }

    try:
        async with httpx.AsyncClient(timeout=config.GOOGLE_SEARCH_TIMEOUT_SECONDS) as client:
            resp = await client.get(
                "https://www.googleapis.com/customsearch/v1", params=params
            )
    except httpx.TimeoutException:
        raise RuntimeError("Google Custom Search request timed out.")
    except httpx.RequestError as e:
        raise RuntimeError(f"Google Custom Search request failed: {e}")

    if resp.status_code == 429:
        raise RuntimeError("Google Custom Search daily quota exceeded.")
    if resp.status_code != 200:
        raise RuntimeError(
            f"Google Custom Search returned {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    items = data.get("items", [])

    return [
        LiveResult(
            title=item.get("title", ""),
            link=item.get("link", ""),
            snippet=item.get("snippet", ""),
        )
        for item in items
    ]


@app.post("/live-search", response_model=LiveSearchResponse)
async def live_search(req: SearchRequest):
    """Check whether a semantically similar question exists live on
    Quora.com right now, independent of our static local dataset."""
    start = time.time()

    query = clean_text(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning.")

    try:
        results = await _google_search_quora(query)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return LiveSearchResponse(
        query=req.query,
        results=results,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/combined-search", response_model=CombinedSearchResponse)
async def combined_search(req: SearchRequest):
    """Run the local FAISS search and the live Quora.com check in
    parallel. If the live check fails (quota, network, not configured),
    local results still come back -- live_search_error explains why."""
    start = time.time()

    model = get_model()
    index = get_index()
    questions = get_questions()

    query = clean_text(req.query)
    if not query:
        raise HTTPException(status_code=400, detail="Query is empty after cleaning.")

    top_k = min(req.top_k, index.ntotal)

    def _local_search():
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=config.DEVICE,
        ).astype(np.float32)
        scores, indices = index.search(query_embedding, top_k)
        return [
            SearchResult(question=questions.iloc[idx]["question"], similarity=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    local_task = asyncio.to_thread(_local_search)
    live_task = _google_search_quora(query)

    local_results, live_outcome = await asyncio.gather(
        local_task, live_task, return_exceptions=True
    )

    if isinstance(local_results, Exception):
        raise HTTPException(status_code=500, detail=str(local_results))

    if isinstance(live_outcome, Exception):
        live_results = []
        live_error = str(live_outcome)
    else:
        live_results = live_outcome
        live_error = None

    return CombinedSearchResponse(
        query=req.query,
        local_results=local_results,
        live_results=live_results,
        live_search_error=live_error,
        latency_ms=round((time.time() - start) * 1000, 2),
    )


@app.post("/reindex")
def reindex():
    """Pick up newly written model/index/corpus files without restarting
    the process. Call this after re-running train_sbert.py + build_index.py."""
    try:
        reset_caches()
        get_model()
        get_index()
        get_questions()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "reindexed"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)