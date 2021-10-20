import dataclasses
from enum import unique
import click
import datasets
from datasets import features
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from src.utils.utils import remove_BIE
import dataclasses
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict
from logging import getLogger
from src.utils.params import span_length

logger = getLogger(__name__)


@dataclasses.dataclass
class SpanClassificationDatasetArgs:
    span_length: int = span_length
    label_balance: bool = False
    hard_o_sampling: bool = False
    o_outside_entity: bool = False
    weight_of_hard_o_for_easy_o: float = 0.5  #


from tqdm import tqdm
from collections import Counter
import random


def get_o_under_sampling_ratio(
    ner_dataset: datasets.Dataset,
    span_classification_dataset_args: SpanClassificationDatasetArgs,
):
    span_length = span_classification_dataset_args.span_length
    ner_tag_labels = ner_dataset.features["ner_tags"].feature.names
    labels = []
    assert len(ner_dataset) > 0
    for snt in tqdm(ner_dataset):
        ner_tags = [ner_tag_labels[tag] for tag in snt["ner_tags"]]
        span2label = {(s, e + 1): label for label, s, e in get_entities(ner_tags)}
        snt_length = len(snt["tokens"])
        span_generator = (
            (i, j)
            for i in range(snt_length)
            for j in range(i + 1, snt_length)
            if j - i <= span_length
        )
        in_ent_words = {wid for s, e in span2label.keys() for wid in range(s, e)}
        for s, e in span_generator:
            if (s, e) in span2label:
                labels.append(span2label[(s, e)])
            else:
                if span_classification_dataset_args.hard_o_sampling:
                    if set(range(s, e)) & in_ent_words:
                        if span_classification_dataset_args.o_outside_entity:
                            if not set(range(s, e)) <= in_ent_words:
                                labels.append("hard_O")
                        else:
                            labels.append("hard_O")
                    else:
                        labels.append("easy_O")
                else:
                    labels.append("O")
        # labels += [
        #     span2label[(s, e)] if (s, e) in span2label else "O"
        #     for s, e in span_generator
        # ]
    if span_classification_dataset_args.hard_o_sampling:
        dataset_count = Counter(labels)
        non_o_label_count = sum(
            v for k, v in dataset_count.items() if k not in {"easy_O", "hard_O"}
        )
        easy_o_label_under_sampling_ratio = (
            (1 - span_classification_dataset_args.weight_of_hard_o_for_easy_o)
            * non_o_label_count
            / dataset_count["easy_O"]
        )
        hard_o_label_under_sampling_ratio = (
            span_classification_dataset_args.weight_of_hard_o_for_easy_o
            * non_o_label_count
            / dataset_count["hard_O"]
        )
        return easy_o_label_under_sampling_ratio, hard_o_label_under_sampling_ratio
    else:
        dataset_count = Counter(labels)
        non_o_label_count = sum(v for k, v in dataset_count.items() if k != "O")
        o_label_under_sampling_ratio = non_o_label_count / dataset_count["O"]
        return o_label_under_sampling_ratio


def ner_datasets_to_span_classification_datasets(
    ner_datasets: datasets.DatasetDict,
    span_classification_dataset_args: SpanClassificationDatasetArgs,
) -> datasets.DatasetDict:
    pre_span_classification_datasets = dict()

    label_names = sorted(
        set(
            [
                remove_BIE(tag)
                for tag in ner_datasets["test"].features["ner_tags"].feature.names
                if tag != "O"
            ]
        )
    )
    info = datasets.DatasetInfo(
        features=datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "starts": datasets.Sequence(datasets.Value("int32")),
                "ends": datasets.Sequence(datasets.Value("int32")),
                "labels": datasets.Sequence(datasets.ClassLabel(names=label_names)),
            }
        )
    )
    for key in ner_datasets:
        pre_span_classification_dataset = defaultdict(list)
        ner_tag_labels = ner_datasets[key].features["ner_tags"].feature.names
        for snt in tqdm(ner_datasets[key]):
            ner_tags = [ner_tag_labels[tag] for tag in snt["ner_tags"]]
            starts = []
            ends = []
            labels = []
            for label, s, e in get_entities(ner_tags):
                starts.append(s)
                ends.append(e + 1)
                labels.append(label)
            if labels:
                pre_span_classification_dataset["tokens"].append(snt["tokens"])
                pre_span_classification_dataset["starts"].append(starts)
                pre_span_classification_dataset["ends"].append(ends)
                pre_span_classification_dataset["labels"].append(labels)
        pre_span_classification_datasets[key] = datasets.Dataset.from_dict(
            pre_span_classification_dataset, info=info
        )
    return datasets.DatasetDict(pre_span_classification_datasets)


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
            if e - s <= SpanClassificationDatasetArgs.span_length
        ]
        for s, e in random.sample(spans, min(span_num_per_snt, len(spans))):
            o_label_spans.append(snt[s:e])
    return random.sample(o_label_spans, min(span_num, len(o_label_spans)))


import spacy


from itertools import islice
from dataclasses import dataclass


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
