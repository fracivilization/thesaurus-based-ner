import collections

from datasets.arrow_dataset import Dataset


class UnionFind:
    def __init__(self):
        self.leaders = collections.defaultdict(lambda: None)

    def find(self, x):
        l = self.leaders[x]
        if l is not None:
            l = self.find(l)
            self.leaders[x] = l
            return l
        return x

    def union(self, x, y):
        lx, ly = self.find(x), self.find(y)
        if lx != ly:
            self.leaders[lx] = ly

    def get_groups(self):
        groups = collections.defaultdict(set)
        for x in self.leaders:
            groups[self.find(x)].add(x)
        return list(groups.values())


def remove_BIE(ner_tag):
    if ner_tag[:2] in {"B-", "I-", "E-"}:
        return ner_tag[2:]
    elif ner_tag == "O":
        return ner_tag
    else:
        raise NotImplementedError


def remove_BIE(ner_tag):
    if ner_tag[:2] in {"B-", "I-", "E-"}:
        return ner_tag[2:]
    elif ner_tag == "O":
        return ner_tag
    else:
        raise NotImplementedError


def remain_specified_data(dataset: Dataset, num: int):
    remained_dataset = dataset[:num]
    return Dataset.from_dict(remained_dataset, features=dataset.features)



