import os
import re
import logging
from functools import lru_cache

import faiss
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def setup_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


logger = setup_logger(__name__)


# --------------------------------------------------------------------------
# Config (env-var driven, sane defaults for local dev)
# --------------------------------------------------------------------------

class Config:
    MODEL_NAME = os.getenv(
        "SBERT_MODEL_NAME",
        "sentence-transformers/all-mpnet-base-v2",
    )
    FINETUNED_MODEL_PATH = os.getenv(
        "FINETUNED_MODEL_PATH",
        "models/sbert-quora-finetuned",
    )
    # Set to "1" to load the fine-tuned model instead of the base model.
    USE_FINETUNED = os.getenv("USE_FINETUNED", "0") == "1"

    INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/embeddings/faiss_index.index")
    QUESTIONS_PATH = os.getenv("UNIQUE_QUESTIONS_PATH", "data/embeddings/questions.csv")
    EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "data/embeddings/question_embeddings.npy")

    RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/raw/questions.csv")
    CLEAN_DATA_PATH = os.getenv("CLEAN_DATA_PATH", "data/processed/clean_questions.csv")

    DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.80"))
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


config = Config()


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Minimal, reversible-ish cleaning. Keep it light -- SBERT handles
    casing/punctuation fine, aggressive cleaning (stopword removal, stemming)
    actively hurts sentence embedding quality."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


# --------------------------------------------------------------------------
# Singleton loaders -- avoid reloading the model / index per request.
# Use lru_cache so repeated calls (e.g. across FastAPI request handlers)
# return the same object instead of re-reading from disk each time.
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model_path = config.FINETUNED_MODEL_PATH if config.USE_FINETUNED else config.MODEL_NAME
    logger.info(f"Loading SentenceTransformer model: {model_path} on {config.DEVICE}")
    model = SentenceTransformer(model_path, device=config.DEVICE)
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_index() -> faiss.Index:
    if not os.path.exists(config.INDEX_PATH):
        raise FileNotFoundError(
            f"FAISS index not found at {config.INDEX_PATH}. Run build_index.py first."
        )
    logger.info(f"Loading FAISS index: {config.INDEX_PATH}")
    return faiss.read_index(config.INDEX_PATH)


@lru_cache(maxsize=1)
def get_questions() -> pd.DataFrame:
    if not os.path.exists(config.QUESTIONS_PATH):
        raise FileNotFoundError(
            f"Questions corpus not found at {config.QUESTIONS_PATH}. Run train_sbert.py first."
        )
    df = pd.read_csv(config.QUESTIONS_PATH)
    if "question" not in df.columns:
        raise ValueError(
            f"Expected a 'question' column in {config.QUESTIONS_PATH}, "
            f"got columns: {list(df.columns)}"
        )
    logger.info(f"Loaded {len(df)} unique questions")
    return df


def reset_caches() -> None:
    """Call after re-running the training/indexing pipeline so the API
    picks up new artifacts without a process restart (used by /reindex)."""
    get_model.cache_clear()
    get_index.cache_clear()
    get_questions.cache_clear()