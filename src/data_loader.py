import os
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


class DataLoader:

    REQUIRED_COLUMNS = [
        "question1",
        "question2",
        "is_duplicate"
    ]

    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):

        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset not found : {self.filepath}"
            )

        logger.info("Loading dataset...")

        self.data = pd.read_csv(self.filepath)

        logger.info(f"Dataset Loaded Successfully.")

        logger.info(f"Shape : {self.data.shape}")

        return self.data

    def validate_columns(self):

        missing = [
            col for col in self.REQUIRED_COLUMNS
            if col not in self.data.columns
        ]

        if len(missing):

            raise ValueError(
                f"Missing Columns : {missing}"
            )

        logger.info("Column Validation Passed.")

    def remove_missing(self):

        before = len(self.data)

        self.data.dropna(
            subset=self.REQUIRED_COLUMNS,
            inplace=True
        )

        after = len(self.data)

        logger.info(
            f"Removed {before-after} rows with missing values."
        )

    def remove_duplicates(self):

        before = len(self.data)

        self.data.drop_duplicates(inplace=True)

        after = len(self.data)

        logger.info(
            f"Removed {before-after} duplicate rows."
        )

    def dataset_info(self):

        logger.info("=" * 60)

        logger.info(f"Rows : {len(self.data)}")

        logger.info(f"Columns : {len(self.data.columns)}")

        logger.info(self.data.dtypes)

        logger.info("=" * 60)

    def label_distribution(self):

        logger.info("Label Distribution")

        logger.info(
            self.data["is_duplicate"].value_counts()
        )

    def save_clean_data(self, output_path):

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        self.data.to_csv(
            output_path,
            index=False
        )

        logger.info(
            f"Clean Dataset Saved -> {output_path}"
        )


if __name__ == "__main__":

    loader = DataLoader(
        "data/raw/questions.csv"
    )

    loader.load_data()

    loader.validate_columns()

    loader.remove_missing()

    loader.remove_duplicates()

    loader.dataset_info()

    loader.label_distribution()

    loader.save_clean_data(
        "data/processed/clean_questions.csv"
    )