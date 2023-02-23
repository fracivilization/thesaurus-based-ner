import collections
from pathlib import Path
from datasets.arrow_dataset import Dataset
from tqdm import tqdm
import dartsclone
import json
from hydra.utils import to_absolute_path


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


class LazyFileList:
    def __init__(self, file_path: Path):
        print("load lazyfile list length")
        with open(file_path) as f:
            self.length = sum(1 for _ in tqdm(f))
        self.file_io = open(file_path)

    def __iter__(self):
        return self

    def __next__(self):
        line = self.file_io.readline().strip()
        if line:
            return line
        else:
            raise StopIteration()

    def __len__(self):
        return self.length


class LazyMapIteratorWithLength:
    """len関数を使えるようなmap関数の代わりのクラス
    遅延評価でメモリを減らしつつ、lenが必要な処理が使えるようにする
    """

    def __init__(self, iterator, func, length):
        self.iterator = iterator.__iter__()
        self.func = func
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        try:
            next_item = next(self.iterator)
        except StopIteration:
            raise StopIteration()
        else:
            return self.func(next_item)

    def __len__(self):
        return self.length


class DoubleArrayDict:
    def __init__(self, sorted_keys, indexed_values, value_labels) -> None:
        """DoubleArrayを利用した辞書

        sorted_keys: 辞書順のbinary化済みkeys
        indexed_values: keysに対応する値(int)
        values_labels: indexed_valuesをもとに戻すためのlist
        """
        self.double_array = dartsclone.DoubleArray()
        self.value_labels = value_labels
        self.double_array.build(sorted_keys, values=indexed_values)

    def __getitem__(self, key: str):
        value_index, _ = self.double_array.exact_match_search(key.encode("utf-8"))
        return self.value_labels[value_index]

    def __contains__(self, item: str):
        value_index, _ = self.double_array.exact_match_search(item.encode("utf-8"))
        return value_index != -1

    def load_from_unsorted_unbinarized_key_and_indexed_values(
        keys, indexed_values, value_labels
    ):
        print("start encoding")
        encoded_keys = [key.encode("utf-8") for key in tqdm(keys)]
        print("start sort")
        encoded_keys, values = zip(
            *sorted(zip(encoded_keys, indexed_values), key=lambda x: x[0])
        )
        values_set = list(sorted(set(values)))
        indexed_values = [values_set.index(val) for val in values]
        return DoubleArrayDict(encoded_keys, indexed_values, value_labels)

    def load_from_jsonl_key_value_pairs(jsonl_key_value_pairs_file_path: Path):
        """jsonlファイル形式の辞書からDoubleArrayDictのロードを行う。

        jsonlファイルは各行がkeyとvalueのペアで構成されている。ただし末端は改行のみ。

        例: ["!!!!!!!", ["<http://dbpedia.org/ontology/Album>", "<http://dbpedia.org/ontology/MusicalWork>"]]
        この場合 "!!!!!!!" がkeyで
        ["<http://dbpedia.org/ontology/Album>", "<http://dbpedia.org/ontology/MusicalWork>"] がvalue
        """
        cats_list = []
        keys = []
        values = []
        for line in tqdm(
            open(to_absolute_path(jsonl_key_value_pairs_file_path)), total=11476181
        ):
            if line:
                term, cats = json.loads(line.strip())
                keys.append(term)
                if cats not in cats_list:
                    cats_list.append(cats)
                values.append(cats_list.index(cats))

        return DoubleArrayDict.load_from_unsorted_unbinarized_key_and_indexed_values(
            keys, values, cats_list
        )
