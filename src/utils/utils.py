import collections
from pathlib import Path
from datasets.arrow_dataset import Dataset
from tqdm import tqdm
import dartsclone
import json
import os


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
        print("calulating key value pairs count")
        total_key_value_count = sum(1 for _ in open(jsonl_key_value_pairs_file_path))
        for line in tqdm(
            open(jsonl_key_value_pairs_file_path),
            total=total_key_value_count,
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


class FileBasedIterator:
    """ファイルから読み取った行をもとにしたイテレーター"""

    def __init__(self, file_path, func_for_each_line) -> None:
        print("start calculating length")
        # self.length = sum(1 for _ in tqdm(open(file_path)))
        self.f = open(file_path, "r")
        self.func_for_each_line = func_for_each_line

    def __iter__(self):
        return self

    def __next__(self):
        try:
            next_line = self.f.readline()
        except StopIteration:
            raise StopIteration()
        else:
            if next_line.strip():
                return self.func_for_each_line(next_line)
            else:
                raise StopIteration()

    # def __len__(self):
    #     return self.length


class DoubleArrayDictWithIterators:
    """Double Array Dictに.keys, .values, .items などの Iteratorを追加する

    jsonl_key_value_paris (各行がkeyとvalueのペアで表されるようなjsonになっているファイル)からロードする"""

    def __init__(
        self, jsonl_key_value_pairs_file_path: Path, da_dict: DoubleArrayDict = None
    ) -> None:
        self.jsonl_key_value_pairs_file = jsonl_key_value_pairs_file_path
        if da_dict:
            self.da_dict = da_dict
        else:
            self.da_dict = DoubleArrayDict.load_from_jsonl_key_value_pairs(
                jsonl_key_value_pairs_file_path
            )

    def __getitem__(self, item: str):
        if item in self.da_dict:
            return self.da_dict[item]
        else:
            raise KeyError

    def __contains__(self, item: str):
        return item in self.da_dict

    def items(self):
        return FileBasedIterator(
            self.jsonl_key_value_pairs_file, lambda line: json.loads(line.strip())
        )

    def keys(self):
        return FileBasedIterator(
            self.jsonl_key_value_pairs_file, lambda line: json.loads(line.strip())[0]
        )

    def values(self):
        return FileBasedIterator(
            self.jsonl_key_value_pairs_file, lambda line: json.loads(line.strip())[1]
        )

    def save_to_disk(self, target_dir: Path):
        os.makedirs(target_dir, exist_ok=True)
        medata_path = os.path.join(target_dir, "metadata.json")
        double_array_path = os.path.join(target_dir, "double_array.dic")
        with open(medata_path, "w") as f:
            json.dump(
                {
                    "jsonl_key_valu_pairs_file": self.jsonl_key_value_pairs_file,
                    "value_labels": self.da_dict.value_labels,
                },
                f,
            )
        self.da_dict.double_array.save(double_array_path)

    def load_from_disk(target_dir: Path):
        medata_path = os.path.join(target_dir, "metadata.json")
        double_array_path = os.path.join(target_dir, "double_array.dic")

        with open(medata_path, "r") as f:
            metadata = json.load(f)

        da_dict = DoubleArrayDict([], [], metadata["value_labels"])
        da_dict.double_array.clear()
        da_dict.double_array.open(double_array_path)

        return DoubleArrayDictWithIterators(
            metadata["jsonl_key_valu_pairs_file"], da_dict
        )
