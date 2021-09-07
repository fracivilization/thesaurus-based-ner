from datasets import Dataset
import colt


@colt.register("HuggingFaceDataset")
class HuggingFaceDataset:
    def __new__(cls, path: str):
        return Dataset.load_from_disk(path)


@colt.register("ConllOutput")
class ConllOutput:
    def __new__(cls, path: str):
        return Dataset.load_from_disk(path)
