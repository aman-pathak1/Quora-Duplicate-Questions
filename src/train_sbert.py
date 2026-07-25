import os
import math

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

from utils import config, setup_logger, clean_text

logger = setup_logger(__name__)


# --------------------------------------------------------------------------
# 1. Build (question1, question2, label) training examples
# --------------------------------------------------------------------------

def load_pairs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean dataset not found: {path}")

    df = pd.read_csv(path)
    required = {"question1", "question2", "is_duplicate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["question1"] = df["question1"].apply(clean_text)
    df["question2"] = df["question2"].apply(clean_text)
    df = df[(df["question1"] != "") & (df["question2"] != "")]

    logger.info(f"Loaded {len(df)} pairs")
    logger.info(f"Label distribution:\n{df['is_duplicate'].value_counts()}")

    return df


def build_input_examples(df: pd.DataFrame) -> list:
    # CosineSimilarityLoss expects a float label in [0, 1], not a class index.
    return [
        InputExample(texts=[row.question1, row.question2], label=float(row.is_duplicate))
        for row in df.itertuples()
    ]


# --------------------------------------------------------------------------
# 2. Fine-tune
# --------------------------------------------------------------------------

def fine_tune(
    train_examples: list,
    dev_examples: list,
    output_path: str,
    epochs: int = 2,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
):
    logger.info("=" * 60)
    logger.info("Fine-tuning SentenceTransformer")
    logger.info("=" * 60)

    device = config.DEVICE
    logger.info(f"Device: {device}")

    model = SentenceTransformer(config.MODEL_NAME, device=device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    dev_sentences1 = [ex.texts[0] for ex in dev_examples]
    dev_sentences2 = [ex.texts[1] for ex in dev_examples]
    dev_labels = [ex.label for ex in dev_examples]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        dev_sentences1,
        dev_sentences2,
        dev_labels,
        name="dev-eval",
    )

    warmup_steps = math.ceil(len(train_dataloader) * epochs * warmup_ratio)
    logger.info(f"Train batches/epoch: {len(train_dataloader)} | Warmup steps: {warmup_steps}")

    os.makedirs(output_path, exist_ok=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup_steps,
        evaluation_steps=max(len(train_dataloader) // 4, 1),
        output_path=output_path,
        save_best_model=True,
        show_progress_bar=True,
    )

    logger.info(f"Fine-tuned model saved to: {output_path}")
    return model


# --------------------------------------------------------------------------
# 3. Build the unique-question corpus + embeddings the API serves against
#
# IMPORTANT: the corpus must be the union of question1 + question2, deduped.
# Embedding only one column (or the raw pairs file) silently drops
# questions the API can never return in /search results.
# --------------------------------------------------------------------------

def build_question_corpus(df: pd.DataFrame) -> pd.DataFrame:
    unique_questions = pd.concat(
        [df["question1"], df["question2"]], ignore_index=True
    ).dropna()

    unique_questions = unique_questions[unique_questions.str.strip() != ""]
    unique_questions = unique_questions.drop_duplicates().reset_index(drop=True)

    corpus = pd.DataFrame({"question": unique_questions})
    logger.info(f"Unique question corpus size: {len(corpus)}")
    return corpus


def encode_corpus(model: SentenceTransformer, corpus: pd.DataFrame) -> np.ndarray:
    logger.info("Encoding unique question corpus...")

    embeddings = model.encode(
        corpus["question"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,  # required for FAISS IndexFlatIP == cosine sim
        show_progress_bar=True,
        batch_size=64,
        device=config.DEVICE,
    ).astype(np.float32)

    logger.info(f"Embeddings shape: {embeddings.shape}")
    return embeddings


def save_corpus_artifacts(corpus: pd.DataFrame, embeddings: np.ndarray):
    os.makedirs(os.path.dirname(config.QUESTIONS_PATH), exist_ok=True)

    corpus.to_csv(config.QUESTIONS_PATH, index=False)
    logger.info(f"Saved question corpus -> {config.QUESTIONS_PATH}")

    np.save(config.EMBEDDINGS_PATH, embeddings)
    logger.info(f"Saved embeddings -> {config.EMBEDDINGS_PATH}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":

    df = load_pairs(config.CLEAN_DATA_PATH)

    train_df, dev_df = train_test_split(
        df, test_size=0.1, random_state=42, stratify=df["is_duplicate"]
    )

    train_examples = build_input_examples(train_df)
    dev_examples = build_input_examples(dev_df)

    model = fine_tune(
        train_examples=train_examples,
        dev_examples=dev_examples,
        output_path=config.FINETUNED_MODEL_PATH,
    )

    # Rebuild the serving corpus + embeddings from the FULL dataset
    # (train+dev) so nothing indexable gets left out of /search.
    corpus = build_question_corpus(df)
    embeddings = encode_corpus(model, corpus)
    save_corpus_artifacts(corpus, embeddings)

    logger.info("=" * 60)
    logger.info("Done. Next: run build_index.py to (re)build the FAISS index,")
    logger.info("or set USE_FINETUNED=1 and call POST /reindex on the running API.")
    logger.info("=" * 60)