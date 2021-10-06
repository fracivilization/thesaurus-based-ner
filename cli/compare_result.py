import sys
from src.error_analysis.compare import CompareNEROutput
from datasets import Dataset

if __name__ == "__main__":
    CompareNEROutput(
        Dataset.load_from_disk(sys.argv[1]), Dataset.load_from_disk(sys.argv[2])
    )
