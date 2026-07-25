import logging

import faiss
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DuplicateQuestionRetriever:

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2"
    ):

        logger.info("=" * 60)
        logger.info("Initializing Duplicate Question Retriever")
        logger.info("=" * 60)

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Running On : {self.device.upper()}"
        )

        logger.info(
            "Loading SentenceTransformer Model..."
        )

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        self.model.eval()

        self.index = None
        self.questions = None

        logger.info(
            "Model Loaded Successfully."
        )

        logger.info("=" * 60)

    def load_resources(
        self,
        index_path,
        question_path
    ):

        logger.info(
            f"Loading FAISS Index : {index_path}"
        )

        self.index = faiss.read_index(
            index_path
        )

        logger.info(
            f"Indexed Questions : {self.index.ntotal}"
        )

        logger.info(
            f"Loading Questions : {question_path}"
        )

        self.questions = pd.read_csv(
            question_path
        )

        logger.info(
            f"Questions Loaded : {len(self.questions)}"
        )

        logger.info("=" * 60)


    def search(
        self,
        query,
        top_k=5
        ):

        logger.info("=" * 60)
        logger.info("Searching Similar Questions")
        logger.info("=" * 60)

        logger.info(
            f"Query : {query}"
        )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            results.append(
                {
                    "question": self.questions.iloc[idx]["question"],
                    "similarity": float(score)
                }
            )

        logger.info(
            f"Retrieved {len(results)} Similar Questions"
        )

        logger.info("=" * 60)

        return results
    
if __name__ == "__main__":

    INDEX_PATH = (
        "data/embeddings/faiss_index.index"
    )

    QUESTION_PATH = (
        "data/embeddings/questions.csv"
    )

    retriever = DuplicateQuestionRetriever()

    retriever.load_resources(
        INDEX_PATH,
        QUESTION_PATH
    )

    while True:

        print("\n" + "=" * 60)

        query = input(
            "Enter your question (type 'exit' to quit): "
        ).strip()

        if query.lower() == "exit":

            print("Goodbye!")
            break

        results = retriever.search(
            query=query,
            top_k=5
        )

        print("\nTop 5 Similar Questions:\n")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{rank}. {result['question']}"
            )

            print(
                f"   Similarity : {result['similarity']:.4f}"
            )

            print("-" * 60)