from datasets import Dataset, DatasetDict
from typing import Dict, List
from seqeval.metrics.sequence_labeling import get_entities


class MSMLCSentence:
    def __init__(self, snt: Dict, label_names: List[str]):
        self.tokens = snt["tokens"]
        self.snt = " ".join(self.tokens)
        self.starts = snt["starts"]
        self.ends = snt["ends"]
        self.labels = [[label_names[label] for label in span] for span in snt["labels"]]
        self.label_names = label_names
        self.spans = [
            (str(s), str(e), " ".join(self.tokens[s:e]), " ".join(ls))
            for s, e, ls in zip(self.starts, self.ends, self.labels)
            if ls != ["nc-O"]
        ]

    def __repr__(self) -> str:
        spans = "\n".join(["\t".join(span) for span in self.spans])
        return "\n".join([self.snt, spans])

    pass


class MSMLCDataset:
    def __init__(self, msmlc_dataset: Dataset):
        self.msmlc_dataset = msmlc_dataset
        self.features = self.msmlc_dataset.features
        self.label_names = self.msmlc_dataset.features["labels"].feature.feature.names

    def __repr__(self) -> str:
        return self.msmlc_dataset.__repr__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return MSMLCSentence(self.msmlc_dataset[key], self.label_names)
        else:
            return self.msmlc_dataset[key]

    def total_span_num(self) -> int:
        return sum(len(self[snt_i].spans) for snt_i in range(len(self.msmlc_dataset)))

    pass


class MSMLCDatasetDict:
    pass

    def __init__(self, msmlc_dataset_dict):
        self.msmlc_dataset_dict = msmlc_dataset_dict

    def __repr__(self) -> str:
        return self.msmlc_dataset_dict.__repr__()

    @classmethod
    def load_from_disk(cls, path):
        return MSMLCDatasetDict(DatasetDict.load_from_disk(path))

    def __getitem__(self, key):
        return MSMLCDataset(self.msmlc_dataset_dict[key])


class NERSentence:
    def __init__(self, snt: Dict, tag_names: List[str]):
        self.tokens = snt["tokens"]
        self.snt = " ".join(self.tokens)
        self.tags = [tag_names[tag] for tag in snt["ner_tags"]]
        self.tag_names = tag_names
        self.spans = [
            (str(s), str(e + 1), " ".join(self.tokens[s : e + 1]), l)
            for l, s, e in get_entities(self.tags)
        ]

    def __repr__(self) -> str:
        spans = "\n".join(["\t".join(span) for span in self.spans])
        return "\n".join([self.snt, spans])

        pass

    pass


class NERDataset:
    def __init__(self, ner_dataset: Dataset):
        self.ner_dataset = ner_dataset
        self.features = self.ner_dataset.features
        self.tag_names = self.ner_dataset.features["ner_tags"].feature.names

    def __repr__(self) -> str:
        return self.ner_dataset.__repr__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return NERSentence(self.ner_dataset[key], self.tag_names)
        else:
            return self.ner_dataset[key]

    def total_span_num(self) -> int:
        return sum(len(self[snt_i].spans) for snt_i in range(len(self.ner_dataset)))

    pass


class NERDatasetDict:
    def __init__(self, ner_dataset_dict):
        self.ner_dataset_dict = ner_dataset_dict

    def __repr__(self) -> str:
        return self.ner_dataset_dict.__repr__()

    @classmethod
    def load_from_disk(cls, path):
        return NERDatasetDict(DatasetDict.load_from_disk(path))

    def __getitem__(self, key):
        return NERDataset(self.ner_dataset_dict[key])


class NERSentence:
    def __init__(self, snt: Dict, tag_names: List[str]):
        self.tokens = snt["tokens"]
        self.snt = " ".join(self.tokens)
        self.tags = [tag_names[tag] for tag in snt["ner_tags"]]
        self.tag_names = tag_names
        self.spans = [
            (str(s), str(e + 1), " ".join(self.tokens[s : e + 1]), l)
            for l, s, e in get_entities(self.tags)
        ]

    def __repr__(self) -> str:
        spans = "\n".join(["\t".join(span) for span in self.spans])
        return "\n".join([self.snt, spans])

        pass

    pass


class NERDataset:
    def __init__(self, ner_dataset: Dataset):
        self.ner_dataset = ner_dataset
        self.features = self.ner_dataset.features
        self.tag_names = self.ner_dataset.features["ner_tags"].feature.names

    def __repr__(self) -> str:
        return self.ner_dataset.__repr__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return NERSentence(self.ner_dataset[key], self.tag_names)
        else:
            return self.ner_dataset[key]

    def total_span_num(self) -> int:
        return sum(len(self[snt_i].spans) for snt_i in range(len(self.ner_dataset)))

    pass


class NERDatasetDict:
    def __init__(self, ner_dataset_dict):
        self.ner_dataset_dict = ner_dataset_dict

    def __repr__(self) -> str:
        return self.ner_dataset_dict.__repr__()

    @classmethod
    def load_from_disk(cls, path):
        return NERDatasetDict(DatasetDict.load_from_disk(path))

    def __getitem__(self, key):
        return NERDataset(self.ner_dataset_dict[key])


class MultiLabelNERSentence:
    def __init__(self, snt: Dict, label_names: List[str]):
        self.tokens = snt["tokens"]
        self.snt = " ".join(self.tokens)
        self.starts = snt["starts"]
        self.ends = snt["ends"]
        self.labels = [[label_names[label] for label in span] for span in snt["labels"]]
        self.label_names = label_names
        self.spans = [
            (str(s), str(e), " ".join(self.tokens[s:e]), " ".join(ls))
            for s, e, ls in zip(self.starts, self.ends, self.labels)
            if ls != ["nc-O"]
        ]

    def __repr__(self) -> str:
        spans = "\n".join(["\t".join(span) for span in self.spans])
        return "\n".join([self.snt, spans])

    pass


class MultiLabelNERDataset:
    def __init__(self, msmlc_dataset: Dataset):
        self.msmlc_dataset = msmlc_dataset
        self.features = self.msmlc_dataset.features
        self.label_names = self.msmlc_dataset.features["labels"].feature.feature.names

    def __repr__(self) -> str:
        return self.msmlc_dataset.__repr__()

    def __getitem__(self, key):
        if isinstance(key, int):
            return MultiLabelNERSentence(self.msmlc_dataset[key], self.label_names)
        else:
            return self.msmlc_dataset[key]

    def total_span_num(self) -> int:
        return sum(len(self[snt_i].spans) for snt_i in range(len(self.msmlc_dataset)))

    pass


class MultiLabelNERDatasetDict:
    def __init__(self, ner_dataset_dict):
        self.ner_dataset_dict = ner_dataset_dict

    def __repr__(self) -> str:
        return self.ner_dataset_dict.__repr__()

    @classmethod
    def load_from_disk(cls, path):
        return MultiLabelNERDatasetDict(DatasetDict.load_from_disk(path))

    def __getitem__(self, key):
        return MultiLabelNERDataset(self.ner_dataset_dict[key])
