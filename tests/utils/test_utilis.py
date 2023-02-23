import unittest
from src.utils.utils import LazyFileList, DoubleArrayDict, DoubleArrayDictWithIterators
import json
from tqdm import tqdm


class TestLazyFileList(unittest.TestCase):
    def test_lazy_file_list(self):
        lazy_file_list = LazyFileList("tests/fixtures/T031")
        assert len(lazy_file_list) == len(list(lazy_file_list))


class TestDoubleArrayDict(unittest.TestCase):
    def test_load_double_array_dict(self):
        keys = [str(i) for i in range(100)]
        encoded_keys = [key.encode("utf-8") for key in keys]
        values = [str(i * 2) for i in range(100)]
        encoded_keys, values = zip(
            *sorted(zip(encoded_keys, values), key=lambda x: x[0])
        )
        values_set = list(sorted(set(values)))
        indexed_values = [values_set.index(val) for val in values]
        da_dict = DoubleArrayDict(encoded_keys, indexed_values, values_set)
        for key, val in zip(encoded_keys, values):
            assert da_dict[key.decode("utf-8")] == val
        assert "not included word" not in da_dict

    def test_load_term2cats_file(self):
        file_path = "tests/fixtures/term2cats"
        cats_list = []
        keys = []
        values = []
        for line in tqdm(open("tests/fixtures/term2cats"), total=11476181):
            if line:
                term, cats = json.loads(line.strip())
                keys.append(term)
                if cats not in cats_list:
                    cats_list.append(cats)
                values.append(cats_list.index(cats))

        da_dict = DoubleArrayDict.load_from_jsonl_key_value_pairs(file_path)
        for key, val in zip(keys, values):
            assert da_dict[key] == cats_list[val]
        assert "not included word" not in da_dict


class TestDoubleArrayDictWithIterators(unittest.TestCase):
    def test_load_double_array_dict(self):
        file_path = "tests/fixtures/term2cats"

        da_dict = DoubleArrayDictWithIterators(file_path)
        for i, (k, v) in enumerate(da_dict.items()):
            print(i, k, v)
            if i == 10:
                break
        for i, k in enumerate(da_dict.keys()):
            print(i, k)
            if i == 10:
                break
        for i, v in enumerate(da_dict.values()):
            print(i, v)
            if i == 10:
                break
