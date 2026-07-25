import os
import time
import logging

import faiss
import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing FAISS Index Builder")
        logger.info("=" * 60)

        self.index = None

    def load_embeddings(
        self,
        embedding_path,
        question_path
    ):

        logger.info(
            f"Loading Embeddings : {embedding_path}"
        )

        embeddings = np.load(
            embedding_path
        ).astype(np.float32)

        logger.info(
            f"Embedding Shape : {embeddings.shape}"
        )

        logger.info(
            f"Loading Questions : {question_path}"
        )

        questions = pd.read_csv(
            question_path
        )

        logger.info(
            f"Total Questions : {len(questions)}"
        )

        return embeddings, questions
       
    def build_index(
        self,
        embeddings
        ):

        logger.info("=" * 60)
        logger.info("Building FAISS Index")
        logger.info("=" * 60)

        start_time = time.time()

        dimension = embeddings.shape[1]

        logger.info(
            f"Embedding Dimension : {dimension}"
        )

        self.index = faiss.IndexFlatIP(
            dimension
        )

        logger.info(
            "Adding Embeddings To Index..."
        )

        self.index.add(
            embeddings
        )

        logger.info(
            f"Total Indexed Vectors : {self.index.ntotal}"
        )

        logger.info(
            f"Index Build Time : {(time.time() - start_time):.2f} seconds"
        )

        logger.info("=" * 60)

        return self.index
    def save_index(
        self,
        index_path
        ):

        logger.info("=" * 60)
        logger.info("Saving FAISS Index")
        logger.info("=" * 60)

        os.makedirs(
            os.path.dirname(index_path),
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            index_path
        )

        logger.info(
            f"FAISS Index Saved : {index_path}"
        )


if __name__ == "__main__":

    EMBEDDING_PATH = (
        "data/embeddings/question_embeddings.npy"
    )

    QUESTION_PATH = (
        "data/embeddings/questions.csv"
    )

    INDEX_PATH = (
        "data/embeddings/faiss_index.index"
    )

    builder = FAISSIndexBuilder()

    embeddings, questions = builder.load_embeddings(
        EMBEDDING_PATH,
        QUESTION_PATH
    )

    builder.build_index(
        embeddings
    )

    builder.save_index(
        INDEX_PATH
    )

    logger.info("=" * 60)
    logger.info(
        "FAISS Index Created Successfully"
    )
    logger.info("=" * 60)