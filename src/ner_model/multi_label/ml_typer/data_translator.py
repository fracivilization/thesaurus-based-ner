import dataclasses
import datasets
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from src.ner_model.chunker.abstract_model import Chunker
import dataclasses
from collections import defaultdict
from logging import getLogger
from hydra.utils import get_original_cwd
from hashlib import md5
import prettytable
from src.ner_model.chunker import ChunkerConfig
from omegaconf import MISSING


logger = getLogger(__name__)


@dataclasses.dataclass
class MSMLCConfig:
    multi_label_ner_dataset: str = MISSING
    output_dir: str = MISSING
    with_o: bool = False
    chunker: ChunkerConfig = ChunkerConfig()
    under_sample: bool = False


from tqdm import tqdm
from collections import Counter
import random


def log_label_ratio(msmlc_datasets: DatasetDict):
    table = prettytable.PrettyTable(["Label", "Count", "Ratio (%)"])
    pass
    train_dataset = msmlc_datasets["train"]
    label_names = train_dataset.features["labels"].feature.feature.names
    c = Counter(
        [label for snt in train_dataset["labels"] for span in snt for label in span]
    )
    label_sum = sum(c.values())
    for lid, count in c.most_common():
        table.add_row([label_names[lid], count, "%.2f" % (100 * count / label_sum)])
    logger.info(table.get_string())


def remove_misguided_fns(starts, ends, labels):
    new_starts, new_ends, new_labels = [], [], []
    misguided_tokens = set()
    for s, e, l in zip(starts, ends, labels):
        if l == "MISGUIDANCE":
            for i in range(s, e):
                misguided_tokens.add(i)
    for s, e, l in zip(starts, ends, labels):
        if l != "MISGUIDANCE":
            if l.startswith("nc"):
                span = set(range(s, e))
                if span & misguided_tokens:
                    continue
            new_starts.append(s)
            new_ends.append(e)
            new_labels.append(l)
    return new_starts, new_ends, new_labels


def undersample_o_span(msml_dataset: Dataset, info):
    o_span_count: int = 0
    non_o_span_count: int = 0
    label_names = msml_dataset.features["labels"].feature.feature.names
    o_id = label_names.index("nc-O")
    for snt in msml_dataset["labels"]:
        for span in snt:
            if o_id in span:
                o_span_count += 1
            else:
                non_o_span_count += 1
    o_under_sampling_ratio = non_o_span_count / o_span_count
    for snt in msml_dataset["labels"]:
        for span in snt:
            if o_id in span:
                o_span_count += 1
            else:
                non_o_span_count += 1
    ret_starts, ret_ends, ret_labels = [], [], []
    # TODO: ラベル比でUndersamplingする
    raise NotImplementedError
    for snt in msml_dataset:
        ret_snt_starts, ret_snt_ends, ret_snt_labels = [], [], []
        for s, e, ls in zip(snt["starts"], snt["ends"], snt["labels"]):
            if o_id in ls and random.random() > o_under_sampling_ratio:
                continue
            ret_snt_starts.append(s)
            ret_snt_ends.append(e)
            ret_snt_labels.append(ls)
        ret_starts.append(ret_snt_starts)
        ret_ends.append(ret_snt_ends)
        ret_labels.append(ret_snt_labels)
    return Dataset.from_dict(
        {
            "tokens": msml_dataset["tokens"],
            "starts": ret_starts,
            "ends": ret_ends,
            "labels": ret_labels,
        },
        info=info,
    )


def multi_label_ner_datasets_to_multi_span_multi_label_classification_datasets(
    multi_label_ner_datasets: datasets.DatasetDict,
    data_args: MSMLCConfig,
    enumerator: Chunker,
) -> datasets.DatasetDict:
    pre_msml_datasets = dict()
    label_names = (
        multi_label_ner_datasets["test"].features["labels"].feature.feature.names
    )
    if data_args.with_o:
        if "nc-O" not in label_names:
            label_names = ["nc-O"] + label_names
        else:
            raise NotImplementedError
    else:
        assert "nc-O" not in label_names
    info = datasets.DatasetInfo(
        features=datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "starts": datasets.Sequence(datasets.Value("int32")),
                "ends": datasets.Sequence(datasets.Value("int32")),
                "labels": datasets.Sequence(
                    datasets.Sequence(datasets.ClassLabel(names=label_names))
                ),
            }
        )
    )
    for key in multi_label_ner_datasets:
        msml_dataset = defaultdict(list)
        for snt in tqdm(multi_label_ner_datasets[key]):
            registered_chunks = set()
            starts = snt["starts"]
            ends = snt["ends"]
            labels = snt["labels"]
            if data_args.with_o:
                # "nc-O" の分だけ1つずらす
                labels = [[label + 1 for label in span] for span in labels]
            registered_chunks = set(zip(starts, ends))
            for s, e in enumerator.predict(snt["tokens"]):
                if (s, e) not in registered_chunks:
                    starts.append(s)
                    ends.append(e)
                    if data_args.with_o:
                        labels.append(["nc-O"])
                    else:
                        labels.append([])
            if labels:
                msml_dataset["tokens"].append(snt["tokens"])
                msml_dataset["starts"].append(starts)
                msml_dataset["ends"].append(ends)
                msml_dataset["labels"].append(labels)
        pre_msml_datasets[key] = datasets.Dataset.from_dict(msml_dataset, info=info)
    if data_args.under_sample:
        for key in {"train", "validation"}:
            pre_msml_datasets[key] = undersample_o_span(
                pre_msml_datasets[key], info=info
            )
    return datasets.DatasetDict(pre_msml_datasets)


import numpy as np


def label_balancing_span_classification_datasets(
    span_classification_datasets: datasets.DatasetDict, o_and_min_label_count_ratio=1
):
    ret_datasets = dict()
    if "test" in span_classification_datasets:
        info = datasets.DatasetInfo(
            features=span_classification_datasets["test"].features
        )
    else:
        info = datasets.DatasetInfo(
            features=span_classification_datasets["train"].features
        )
    for split_key, dataset_split in span_classification_datasets.items():
        if split_key != "test":
            if "labels" in dataset_split.features:
                # for multi span classification datasets
                span_classification_dataset = {
                    "tokens": [],
                    "starts": [],
                    "ends": [],
                    "labels": [],
                }
                label_count = Counter(
                    [l for snt in dataset_split["labels"] for l in snt]
                )
                min_label_count = min(label_count.values())
                logger.info("min label count: %d" % min_label_count)
                undersampling_ratio = {
                    label: min_label_count / count
                    for label, count in label_count.items()
                }
                for snt in tqdm(dataset_split):
                    starts = []
                    ends = []
                    labels = []
                    for s, e, l in zip(snt["starts"], snt["ends"], snt["labels"]):
                        if random.random() < undersampling_ratio[l]:
                            starts.append(s)
                            ends.append(e)
                            labels.append(l)
                    if labels:
                        span_classification_dataset["tokens"].append(snt["tokens"])
                        span_classification_dataset["starts"].append(starts)
                        span_classification_dataset["ends"].append(ends)
                        span_classification_dataset["labels"].append(labels)
                ret_datasets[split_key] = datasets.Dataset.from_dict(
                    span_classification_dataset, info=info
                )
            elif "label" in dataset_split.features:
                # for one span classification datasets
                span_classification_dataset = {
                    "tokens": [],
                    "start": [],
                    "end": [],
                    "label": [],
                }
                label_names = dataset_split.features["label"].names
                label_count = Counter(dataset_split["label"])
                min_label_count = min(label_count.values())
                logger.info("min label count: %d" % min_label_count)
                undersampling_ratio = dict()
                for label, count in label_count.items():
                    if label_names[label] == "O":
                        undersampling_ratio[label] = (
                            min_label_count / count * o_and_min_label_count_ratio
                        )
                    else:
                        undersampling_ratio[label] = min_label_count / count
                for snt in tqdm(dataset_split):
                    if random.random() < undersampling_ratio[snt["label"]]:
                        for key, value in snt.items():
                            span_classification_dataset[key].append(value)
                ret_datasets[split_key] = datasets.Dataset.from_dict(
                    span_classification_dataset, info=info
                )
            else:
                raise NotImplementedError
        else:
            ret_datasets[split_key] = dataset_split
    return datasets.DatasetDict(ret_datasets)


import os
from pathlib import Path


def print_label_statistics(span_classification_datasets: datasets.DatasetDict):
    for split_key, dataset_split in span_classification_datasets.items():
        if "label" in dataset_split.features:
            label_names = dataset_split.features["label"].names
            label_count = Counter([label_names[l] for l in dataset_split["label"]])
        else:
            pass
            label_names = dataset_split.features["labels"].feature.names
            label_count = Counter(
                [label_names[l] for snt in dataset_split["labels"] for l in snt]
            )
        logger.info("label count of %s split: %s" % (split_key, label_count))


from copy import deepcopy


from typing import Dict, List

import random


def load_o_label_spans(unlabelled_corpus: Dataset, span_num: int) -> List:
    # 各文から取得するスパン数を指定
    # 各文に対してspan_length長のスパンをかき集めてくる
    # 各文に定められた個数になるまでサンプリング
    # 全体の断片から決められたスパン数になるまでサンプリング
    pass
    snt_num = len(unlabelled_corpus)
    span_num_per_snt = int(span_num / snt_num) + 100
    o_label_spans = []
    for snt in unlabelled_corpus["tokens"]:
        spans = [
            (s, e)
            for s in range(len(snt))
            for e in range(s + 1, len(snt) + 1)
            if e - s <= MSMLCConfig.span_length
        ]
        for s, e in random.sample(spans, min(span_num_per_snt, len(spans))):
            o_label_spans.append(snt[s:e])
    return random.sample(o_label_spans, min(span_num, len(o_label_spans)))


import spacy


from itertools import islice
from dataclasses import MISSING, dataclass


@dataclass
class Term2CatBasedDatasetArgs:
    label_balance: bool = False
    pass


def load_term2cat_based_span_classification_dataset(
    term2cat: Dict, unlabelled_corpus: Dataset, args: Term2CatBasedDatasetArgs
):
    tokenizer = spacy.load("en_core_sci_sm")
    tokenizer.remove_pipe("ner")
    dataset = {"tokens": [], "start": [], "end": [], "label": []}
    label_names = ["O"] + sorted(set(term2cat.values()))
    dict_label_count = Counter(term2cat.values())
    if args.label_balance:
        over_sampling_ratio = {
            l: dict_label_count.most_common()[0][1] / dict_label_count[l]
            for l in dict_label_count
        }
    else:
        over_sampling_ratio = {l: 1 for l in dict_label_count}

    for term, cat in tqdm(term2cat.items()):
        osr = over_sampling_ratio[cat]
        tokenized_terms = tokenizer(term)
        while True:
            if 0 < osr < 1:
                if osr > random.random():
                    break
            elif osr <= 0:
                break
            dataset["tokens"].append([w.text for w in tokenized_terms])
            dataset["start"].append(0)
            dataset["end"].append(len(tokenized_terms))
            dataset["label"].append(label_names.index(cat))
            osr -= 1

    if args.label_balance:
        span_num = dict_label_count.most_common()[0][1]
    else:
        span_num = sum(dict_label_count.values())
    o_labeled_spans = load_o_label_spans(unlabelled_corpus, span_num)
    for span in o_labeled_spans:
        dataset["tokens"].append(span)
        dataset["start"].append(0)
        dataset["end"].append(len(span))
        dataset["label"].append(label_names.index("O"))
    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "start": datasets.Value("int32"),
            "end": datasets.Value("int32"),
            "label": datasets.ClassLabel(names=label_names),
        }
    )
    # new_dataset_dictに追加
    return Dataset.from_dict(dataset, features=features)


def split_span_classification_dataset(datasets: Dataset):
    features = datasets.features
    split_num = int(len(datasets) * 0.9)
    splitted_datasets = dict()
    from random import shuffle

    indexes = list(range(len(datasets)))
    shuffle(indexes)

    splitted_datasets["train"] = Dataset.from_dict(
        datasets.__getitem__(indexes[:split_num]), features=features
    )
    splitted_datasets["validation"] = Dataset.from_dict(
        datasets.__getitem__(indexes[split_num:]), features=features
    )
    return DatasetDict(splitted_datasets)


def join_span_classification_datasets(
    main_datasets: DatasetDict, sub_datasets: DatasetDict
):
    pass
    new_dataset_dict = dict()
    for key, split in main_datasets.items():
        if key in sub_datasets:
            sub_split = sub_datasets[key]
            new_dataset = {feature: split[feature] for feature in split.features}
            main_label_names = split.features["label"].names
            sub_label_names = sub_split.features["label"].names
            assert len(main_label_names) == len(sub_label_names)
            assert len(split.features) == len(sub_split.features)
            label_map = {
                i: sub_label_names.index(l) for i, l in enumerate(main_label_names)
            }
            for feature in sub_split.features:
                if feature == "label":
                    new_dataset[feature] += [label_map[l] for l in sub_split[feature]]
                else:
                    new_dataset[feature] += sub_split[feature]
            new_dataset_dict[key] = Dataset.from_dict(new_dataset, split.features)
        else:
            new_dataset_dict[key] = split
    return DatasetDict(new_dataset_dict)


def translate_into_msc_datasets(
    ner_datasets: DatasetDict,
    msc_args: MSMLCConfig,
    enumerator: Chunker,
):
    input_hash = {k: v._fingerprint for k, v in ner_datasets.items()}
    input_hash["msc_args"] = str(msc_args)
    input_hash["enumerator"] = str(enumerator.config)
    output_dir = Path(get_original_cwd()).joinpath(
        "data", "buffer", md5(str(input_hash).encode()).hexdigest()
    )
    logger.info("output_dir of msc_datasets: " + str(output_dir))
    if not output_dir.exists():
        msc_datasets = (
            multi_label_ner_datasets_to_multi_span_multi_label_classification_datasets(
                ner_datasets, msc_args, enumerator
            )
        )
        msc_datasets.save_to_disk(output_dir)
    else:
        msc_datasets = DatasetDict.load_from_disk(output_dir)
    log_label_ratio(msc_datasets)
    return msc_datasets
