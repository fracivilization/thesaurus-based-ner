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

logger = getLogger(__name__)


@dataclasses.dataclass
class SpanClassificationDatasetArgs:
    span_length: int = 6
    span_per_snt: click.Choice(["multiple", "one"]) = "one"
    label_balance: bool = False
    remove_fake_cat: bool = False
    remove_o: bool = False
    hard_o_sampling: bool = False
    o_outside_entity: bool = False
    weight_of_hard_o_for_easy_o: float = 0.5  #
    pre_one_span: bool = False  # if True, make pre_span_classification datasets having one span per one instance


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


def ner_datasets_to_pre_span_classification_datasets(
    ner_datasets: datasets.DatasetDict,
    span_classification_dataset_args: SpanClassificationDatasetArgs,
) -> datasets.DatasetDict:
    pre_span_classification_datasets = dict()

    label_names = ["O"] + sorted(
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
        pass
        pre_span_classification_dataset = defaultdict(list)
        ner_tag_labels = ner_datasets[key].features["ner_tags"].feature.names
        if span_classification_dataset_args.hard_o_sampling:
            (
                easy_o_under_sampling_ratio,
                hard_o_under_sampling_ratio,
            ) = get_o_under_sampling_ratio(
                ner_datasets[key], span_classification_dataset_args
            )

        else:
            o_under_sampling_ratio = get_o_under_sampling_ratio(
                ner_datasets[key], span_classification_dataset_args
            )
        for snt in tqdm(ner_datasets[key]):
            ner_tags = [ner_tag_labels[tag] for tag in snt["ner_tags"]]
            span2label = {(s, e + 1): label for label, s, e in get_entities(ner_tags)}
            words_in_ents = set(i for s, e in span2label.keys() for i in range(s, e))
            snt_length = len(snt["tokens"])
            span_generator = (
                (i, j)
                for i in range(snt_length)
                for j in range(i + 1, snt_length)
                if j - i <= span_classification_dataset_args.span_length
            )
            starts = []
            ends = []
            labels = []
            for s, e in span_generator:
                if key != "test" and (s, e) not in span2label:
                    if span_classification_dataset_args.hard_o_sampling:
                        if (s, e) not in span2label:
                            if set(range(s, e)) & words_in_ents == set():
                                if span_classification_dataset_args.o_outside_entity:
                                    if set(range(s, e)) <= words_in_ents:
                                        continue
                                if random.random() > hard_o_under_sampling_ratio:
                                    continue
                            else:
                                if random.random() > easy_o_under_sampling_ratio:
                                    continue
                    else:
                        if random.random() > o_under_sampling_ratio:
                            continue
                starts.append(s)
                ends.append(e)
                labels.append(span2label[(s, e)] if (s, e) in span2label else "O")
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


def make_one_span_per_snt(pre_span_classification_datasets: DatasetDict):
    """スパンを文ごとに一つになるように修正する

    Args:
        pre_span_classification_datasets (DatasetDict): span classification dataset which include multiple spans
    """
    pass
    label_names = (
        pre_span_classification_datasets["test"].features["labels"].feature.names
    )
    info = datasets.DatasetInfo(
        features=datasets.Features(
            {
                "tokens": datasets.Sequence(datasets.Value("string")),
                "start": datasets.Value("int32"),
                "end": datasets.Value("int32"),
                "label": datasets.ClassLabel(names=label_names),
            }
        )
    )
    span_classification_datasets = dict()
    for split_key, dataset_split in pre_span_classification_datasets.items():
        span_classification_dataset = defaultdict(list)
        for snt in dataset_split:
            tokens = snt["tokens"]
            for s, e, l in zip(snt["starts"], snt["ends"], snt["labels"]):
                span_classification_dataset["tokens"].append(tokens)
                span_classification_dataset["start"].append(s)
                span_classification_dataset["end"].append(e)
                span_classification_dataset["label"].append(l)
        span_classification_datasets[split_key] = datasets.Dataset.from_dict(
            span_classification_dataset, info=info
        )
    return datasets.DatasetDict(span_classification_datasets)


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


def span_classification_datasets_to_span_detection_datasets(
    span_classification_datasets: datasets.DatasetDict,
):
    ret_datasets = dict()
    features = deepcopy(span_classification_datasets["test"].features)
    old_names = features["label"].names
    new_names = ["O", "Span"]
    label_map = dict()
    for i, l in enumerate(old_names):
        if l == "O":
            label_map[i] = new_names.index("O")
        else:
            label_map[i] = new_names.index("Span")
    features["label"] = datasets.ClassLabel(names=new_names)
    info = datasets.DatasetInfo(features=features)
    for split_key, dataset_split in span_classification_datasets.items():
        if "labels" in dataset_split.features:
            raise NotImplementedError
            # for multi span classification datasets
            span_classification_dataset = {
                "tokens": [],
                "starts": [],
                "ends": [],
                "labels": [],
            }
            label_count = Counter([l for snt in dataset_split["labels"] for l in snt])
            min_label_count = min(label_count.values())
            logger.info("min label count: %d" % min_label_count)
            undersampling_ratio = {
                label: min_label_count / count for label, count in label_count.items()
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
                    span_classification_dataset["starts"].append(snt["starts"])
                    span_classification_dataset["ends"].append(snt["ends"])
                    span_classification_dataset["labels"].append(snt["labels"])
            ret_datasets[split_key] = datasets.Dataset.from_dict(
                span_classification_dataset, info=info
            )
        elif "label" in dataset_split.features:
            # for one span classification datasets
            span_classification_dataset = {
                "tokens": dataset_split["tokens"],
                "start": dataset_split["start"],
                "end": dataset_split["end"],
                "label": [],
            }
            for label in tqdm(dataset_split["label"]):
                span_classification_dataset["label"].append(label_map[label])
            ret_datasets[split_key] = datasets.Dataset.from_dict(
                span_classification_dataset, info=info
            )
        else:
            raise NotImplementedError
    return datasets.DatasetDict(ret_datasets)


def remove_fake_cat_from_span_classification_dataset(datasets: DatasetDict):
    names = datasets["train"].features["label"].names
    fake_cat_nums = {i for i, n in enumerate(names) if "fake_cat_" in n}
    for split in datasets.values():
        assert names == split.features["label"].names

    new_datasets = dict()
    for key, split in datasets.items():
        new_snts = defaultdict(list)
        for snt in tqdm(split):
            if snt["label"] not in fake_cat_nums:
                for k, v in snt.items():
                    new_snts[k].append(v)
        new_datasets[key] = Dataset.from_dict(new_snts, features=split.features)
    return DatasetDict(new_datasets)


def remove_o_label_dataset(datasets: DatasetDict):
    names = datasets["train"].features["label"].names
    o_label_id = names.index("O")

    for split in datasets.values():
        assert names == split.features["label"].names

    new_datasets = dict()
    for key, split in datasets.items():
        new_snts = defaultdict(list)
        for snt in tqdm(split):
            if snt["label"] != o_label_id:
                for k, v in snt.items():
                    new_snts[k].append(v)
        new_datasets[key] = Dataset.from_dict(new_snts, features=split.features)
    return DatasetDict(new_datasets)


def pre_one_span_classification_datasets(
    pre_span_classification_datasets: DatasetDict,
) -> DatasetDict:
    pass
    new_dataset_dict = dict()
    for key, split in pre_span_classification_datasets.items():
        new_dataset = defaultdict(list)
        if key not in {"supervised_validation", "test"}:
            for snt in split:
                for s, e, l in zip(snt["starts"], snt["ends"], snt["labels"]):
                    new_dataset["tokens"].append(snt["tokens"])
                    new_dataset["starts"].append([s])
                    new_dataset["ends"].append([e])
                    new_dataset["labels"].append([l])
            new_dataset_dict[key] = Dataset.from_dict(
                new_dataset, features=split.features
            )
        else:
            new_dataset_dict[key] = split
    return DatasetDict(new_dataset_dict)


def ner_datasets_to_span_classification_datasets(
    ner_datasets: datasets.DatasetDict,
    span_classification_dataset_args: SpanClassificationDatasetArgs,
    buffer_dir: str,
) -> datasets.DatasetDict:
    buffer_dir: Path = Path(buffer_dir)
    if not buffer_dir.exists():
        os.mkdir(buffer_dir)
    buffer_dir.joinpath("pre_span_classification_datasets")

    output_path = buffer_dir.joinpath("pre_span_classification_datasets")
    if not output_path.exists():
        pre_span_classification_datasets = (
            ner_datasets_to_pre_span_classification_datasets(
                ner_datasets, span_classification_dataset_args
            )
        )
        pre_span_classification_datasets.save_to_disk(output_path)
    pre_span_classification_datasets = DatasetDict.load_from_disk(output_path)

    output_path = buffer_dir.joinpath("pre_one_span_classification_datasets")
    if span_classification_dataset_args.pre_one_span:
        if not output_path.exists():
            pre_span_classification_datasets = pre_one_span_classification_datasets(
                pre_span_classification_datasets
            )
            pass
        pre_span_classification_datasets.save_to_disk(output_path)

    output_path = buffer_dir.joinpath("span_classification_datasets")
    if span_classification_dataset_args.span_per_snt == "one":
        if not output_path.exists():
            span_classification_datasets = make_one_span_per_snt(
                pre_span_classification_datasets
            )
            span_classification_datasets.save_to_disk(output_path)
    else:
        span_classification_datasets = pre_span_classification_datasets
        span_classification_datasets.save_to_disk(output_path)
    span_classification_datasets = DatasetDict.load_from_disk(output_path)
    logger.info("Without balancing, dataset label statistics: ")
    print_label_statistics(span_classification_datasets)

    output_path = buffer_dir.joinpath("span_classification_datasets_removed_fake_cat")
    if span_classification_dataset_args.remove_fake_cat:
        if not output_path.exists():
            span_classification_datasets = (
                remove_fake_cat_from_span_classification_dataset(
                    span_classification_datasets
                )
            )
            span_classification_datasets.save_to_disk(output_path)
        span_classification_datasets = DatasetDict.load_from_disk(output_path)
        logger.info("Fake categories are removed, dataset label statistics: ")
        print_label_statistics(span_classification_datasets)

    output_path = buffer_dir.joinpath("span_classification_datasets_o_removed")
    if span_classification_dataset_args.remove_o:
        if not output_path.exists():
            span_classification_datasets = remove_o_label_dataset(
                span_classification_datasets
            )
            span_classification_datasets.save_to_disk(output_path)
        span_classification_datasets = DatasetDict.load_from_disk(output_path)
        logger.info("O label are removed, dataset label statistics: ")
        print_label_statistics(span_classification_datasets)

    output_path = buffer_dir.joinpath("span_detection_classification_datasets")
    if span_classification_dataset_args.span_detection:
        if not output_path.exists():
            span_detection_datasets = (
                span_classification_datasets_to_span_detection_datasets(
                    span_classification_datasets
                )
            )
            span_detection_datasets.save_to_disk(output_path)
            logger.info("With balancing, dataset label statistics: ")
        span_detection_datasets = DatasetDict.load_from_disk(output_path)
        span_classification_datasets = span_detection_datasets
        print_label_statistics(span_classification_datasets)

    output_path = buffer_dir.joinpath("span_classification_balanced_datasets")
    if span_classification_dataset_args.label_balance:
        if not output_path.exists():
            span_classification_balanced_datasets = (
                label_balancing_span_classification_datasets(
                    span_classification_datasets
                )
            )
            span_classification_balanced_datasets.save_to_disk(output_path)
            logger.info("With balancing, dataset label statistics: ")
        span_classification_balanced_datasets = DatasetDict.load_from_disk(output_path)
        span_classification_datasets = span_classification_balanced_datasets
        print_label_statistics(span_classification_datasets)

    return span_classification_datasets


from typing import Dict, List
from lib.data.span_classification import SpanClassificationDatasetArgs

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
