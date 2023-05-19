import collections
from pathlib import Path
from datasets.arrow_dataset import Dataset
from tqdm import tqdm
import dartsclone
import json
import os
import shutil
import numpy as np
import copy
import sqlite3
import tempfile
from dataclasses import dataclass
from typing import List


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

    def save_to_disk(self, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        medata_path = os.path.join(target_dir, "metadata.json")
        with open(medata_path, "w") as f:
            json.dump(
                {
                    "value_labels": self.value_labels,
                },
                f,
            )
        double_array_path = os.path.join(target_dir, "double_array.dic")
        self.double_array.save(double_array_path)

    def load_from_disk(target_dir):
        medata_path = os.path.join(target_dir, "metadata.json")
        with open(medata_path, "r") as f:
            metadata = json.load(f)

        da_dict = DoubleArrayDict([], [], metadata["value_labels"])
        da_dict.double_array.clear()
        double_array_path = os.path.join(target_dir, "double_array.dic")
        da_dict.double_array.open(double_array_path)
        return da_dict


class FileBasedIterator:
    """ファイルから読み取った行をもとにしたイテレーター"""

    def __init__(self, file_path, func_for_each_line) -> None:
        # print("start calculating length")
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
        source_kv_pairs_path = os.path.join(target_dir, "source_key_value_pairs.jsnol")
        shutil.copyfile(self.jsonl_key_value_pairs_file, source_kv_pairs_path)
        double_array_dir = os.path.join(target_dir, "double_array")
        self.da_dict.save_to_disk(double_array_dir)

    def load_from_disk(target_dir: Path):
        source_kv_pairs_path = os.path.join(target_dir, "source_key_value_pairs.jsnol")
        double_array_dir = os.path.join(target_dir, "double_array")
        da_dict = DoubleArrayDict.load_from_disk(double_array_dir)

        return DoubleArrayDictWithIterators(source_kv_pairs_path, da_dict)


class SQliteJsonDict:
    def __init__(
        self,
        db_file_path: Path,
        commit_when_set_item: bool = True,
    ) -> None:
        self.db_file_path = db_file_path
        self.commit_when_set_item = commit_when_set_item
        self.con = sqlite3.connect(db_file_path)
        cur = self.con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS key_value (key text, value text)")
        cur.execute("CREATE INDEX IF NOT EXISTS key_index ON key_value (key)")
        self.con.commit()

        cur.close()

    def save_to_disk(self, target_path: Path):
        if self.db_file_path == target_path:
            self.commit()
        else:
            shutil.copyfile(self.db_file_path, target_path)

    def load_from_disk(target_path: Path):
        return SQliteJsonDict(target_path)

    def items(self):
        cur = self.con.cursor()
        chunk_size = 100000
        last_key = None
        while True:
            if last_key:
                cur.execute(
                    "SELECT key, value FROM key_value WHERE key > ? ORDER BY key LIMIT ?",
                    (last_key, chunk_size),
                )
            else:
                cur.execute(
                    "SELECT key, value FROM key_value ORDER BY key LIMIT ?",
                    (chunk_size,),
                )
            rows = cur.fetchall()
            if not rows:
                break

            for key, value in rows:
                yield json.loads(key), json.loads(value)
            last_key = key
        cur.close()

    def keys(self):
        cur = self.con.cursor()
        chunk_size = 100000
        offset = 0
        while True:
            cur.execute(
                "SELECT key FROM key_value LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for key in rows:
                yield json.loads(key)

            offset += chunk_size
        cur.close()

    def values(self):
        cur = self.con.cursor()
        cur.execute("SELECT value FROM key_value_weight_triples")
        for value in cur.fetchall():
            yield json.loads(value)

    def __setitem__(self, key, value):
        cur = self.con.cursor()
        cur.execute(
            "INSERT INTO key_value VALUES (?, ?)",
            (json.dumps(key), json.dumps(value)),
        )
        if self.commit_when_set_item:
            self.con.commit()

    def __getitem__(self, key):
        cur = self.con.cursor()
        cur.execute(
            "SELECT value FROM key_value WHERE key = ?",
            (json.dumps(key),),
        )
        value = cur.fetchone()[0]
        return json.loads(value)

    def __contains__(self, key: str):
        cur = self.con.cursor()
        cur.execute(
            "SELECT value FROM key_value WHERE key = ?",
            (json.dumps(key),),
        )
        return cur.fetchone() is not None

    def commit(self):
        self.con.commit()

    def __len__(self):
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM key_value")
        return cur.fetchone()[0]


@dataclass
class WeightedValues:
    values: List[str]
    weights: List[float]


class WeightedSQliteDict:
    def __init__(
        self,
        db_file_path: Path,
        commit_when_set_item: bool = True,
    ) -> None:
        self.db_file_path = db_file_path
        self.commit_when_set_item = commit_when_set_item
        self.con = sqlite3.connect(db_file_path)
        cur = self.con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS key_value_weight_triples (key text, value text, weight text)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS key_index ON key_value_weight_triples (key)"
        )
        self.con.commit()

        cur.close()

    def load_from_jsonl_key_value_weight_triples_file(
        jsonl_key_value_weight_triples_file_path: Path,
    ):
        work_dir = tempfile.TemporaryDirectory()
        db_file_path = os.path.join(work_dir.name, "db.sqlite3")
        con = sqlite3.connect(db_file_path)
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS key_value_weight_triples (key text, value text, weight text)"
        )
        total_key_value_count = 0
        for line in tqdm(open(jsonl_key_value_weight_triples_file_path)):
            total_key_value_count += 1
            if line:
                term, values, weights = json.loads(line.strip())
                cur.execute(
                    "INSERT INTO key_value_weight_triples VALUES (?, ?, ?)",
                    (term, json.dumps(values), json.dumps(weights)),
                )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS key_index ON key_value_weight_triples (key)"
        )
        con.commit()

        cur.close()
        con.close()
        this = WeightedSQliteDict(db_file_path)
        this.work_dir = work_dir
        return this

    def save_to_disk(self, target_path: Path):
        if self.db_file_path == target_path:
            self.commit()
        else:
            shutil.copyfile(self.db_file_path, target_path)

    def load_from_disk(target_path: Path):
        return WeightedSQliteDict(target_path)

    def items(self):
        cur = self.con.cursor()
        chunk_size = 100000
        last_key = None
        while True:
            if last_key:
                cur.execute(
                    "SELECT key, value, weight FROM key_value_weight_triples WHERE key > ? ORDER BY key LIMIT ?",
                    (last_key, chunk_size),
                )
            else:
                cur.execute(
                    "SELECT key, value, weight FROM key_value_weight_triples ORDER BY key LIMIT ?",
                    (chunk_size,),
                )
            rows = cur.fetchall()
            if not rows:
                break

            for key, value, weight in rows:
                yield key, WeightedValues(json.loads(value), json.loads(weight))
            last_key = key
        cur.close()

    def keys(self):
        cur = self.con.cursor()
        chunk_size = 100000
        offset = 0
        while True:
            cur.execute(
                "SELECT key FROM key_value_weight_triples LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for key in rows:
                yield key[0]

            offset += chunk_size
        cur.close()

    def values(self):
        cur = self.con.cursor()
        cur.execute("SELECT value, weight FROM key_value_weight_triples")
        for value, weight in cur.fetchall():
            yield WeightedValues(json.loads(value), json.loads(weight))

    def __iter__(self):
        return self.keys()

    def __setitem__(self, key: str, value: WeightedValues):
        assert isinstance(value, WeightedValues)
        cur = self.con.cursor()
        cur.execute(
            "INSERT INTO key_value_weight_triples VALUES (?, ?, ?)",
            (key, json.dumps(value.values), json.dumps(value.weights)),
        )
        if self.commit_when_set_item:
            self.con.commit()

    def __getitem__(self, key: str) -> WeightedValues:
        cur = self.con.cursor()
        cur.execute(
            "SELECT value, weight FROM key_value_weight_triples WHERE key = ?",
            (key,),
        )
        value, weight = cur.fetchone()
        return WeightedValues(json.loads(value), json.loads(weight))

    def __contains__(self, key: str):
        cur = self.con.cursor()
        cur.execute(
            "SELECT value, weight FROM key_value_weight_triples WHERE key = ?",
            (key,),
        )
        return cur.fetchone() is not None

    def commit(self):
        self.con.commit()

    def __len__(self):
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM key_value_weight_triples")
        return cur.fetchone()[0]
