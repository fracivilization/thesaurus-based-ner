from datasets import Dataset, DatasetDict
from datasets.info import DatasetInfo
from seqeval.metrics.sequence_labeling import get_entities
import datasets
from datasets import Dataset, DatasetDict
from src.ner_model.multi_label.abstract_model import (
    MultiLabelNERModel,
    MultiLabelNERModelConfig,
)
from dataclasses import dataclass
from logging import getLogger
from tqdm import tqdm
import json
import yaml
from omegaconf import OmegaConf
from typing import List
from src.ner_model.multi_label.two_stage import MultiLabelTwoStageConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig
from src.ner_model.multi_label.ml_typer.dict_match import MultiLabelDictMatchTyperConfig
from omegaconf import MISSING

logger = getLogger(__name__)


@dataclass
class PseudoMSMLCAnnoConfig:
    multi_label_ner_model: MultiLabelNERModelConfig = MultiLabelTwoStageConfig(
        chunker=SpacyNPChunkerConfig(),
        multi_label_typer=MultiLabelDictMatchTyperConfig(),
    )
    output_dir: str = MISSING
    raw_corpus: str = MISSING
    gold_corpus: str = MISSING


def remove_fp_ents(pred_tags: List[str], gold_tags: List[str]):
    new_tags = ["O"] * len(pred_tags)
    for pred_label, s, e in get_entities(pred_tags):
        remain_flag = False
        if pred_label.startswith("nc-"):
            remain_flag = True
        else:
            partially_match_labels = [
                l for l, s, e in get_entities(gold_tags[s : e + 1])
            ]
            if pred_label in partially_match_labels:
                remain_flag = True
        if remain_flag:
            for i in range(s, e + 1):
                if i == s:
                    new_tags[i] = "B-%s" % pred_label
                else:
                    new_tags[i] = "I-%s" % pred_label
    return new_tags


def add_dict_erosion_entity(pred_tags, gold_tags):
    """Add dictionary erosion entity to the prediction tags.
    Dictionry erosion entity is a entity which is in gold tags but not in pred_tags.

    In this research, we define matching between gold and predicted entities as ends with match.
    So, we add gold entities which have no partial match entities with predicted entities.
    e.g.
        pred_tags: B-T038 I-T038 I-T038 I-T038 I-T038 O
        gold_tags: O B-T082 I-T082 I-T082 B-T038 O
        output: B-T038 I-T082 I-T082 I-T082 I-T038 O
    """
    new_tags = copy.deepcopy(pred_tags)
    for gold_label, s, e in get_entities(gold_tags):
        predicted_labels = set(pl for pl, ps, pe in get_entities(pred_tags[s : e + 1]))
        if gold_label not in predicted_labels:
            for i in range(s, e + 1):
                if i == s:
                    new_tags[i] = "B-%s" % gold_label
                else:
                    new_tags[i] = "I-%s" % gold_label
    # TODO: 編集距離も追加する
    raise NotImplementedError
    return new_tags


def get_msml_dataset_features(label_names, with_weitht: False):
    features = {
        "tokens": datasets.Sequence(datasets.Value("string")),
        "starts": datasets.Sequence(datasets.Value("int32")),
        "ends": datasets.Sequence(datasets.Value("int32")),
        "labels": datasets.Sequence(
            datasets.Sequence(datasets.ClassLabel(names=label_names))
        ),
    }
    if with_weitht:
        features["weights"] = datasets.Sequence(
            datasets.Sequence(datasets.Value("float32"))
        )
    features = datasets.Features(features)
    return features


def load_msml_pseudo_dataset(
    raw_corpus: Dataset,
    multi_label_ner_model: MultiLabelNERModel,
    conf: PseudoMSMLCAnnoConfig,
) -> Dataset:
    desc = dict()
    desc["raw_corpus"] = json.loads(raw_corpus.info.description)
    desc["multi_label_ner_model"] = yaml.safe_load(
        OmegaConf.to_yaml(multi_label_ner_model.conf)
    )

    tokens = raw_corpus["tokens"]
    starts, ends, outputs = multi_label_ner_model.batch_predict(tokens)
    label_names = multi_label_ner_model.label_names
    ret_tokens, ret_starts, ret_ends, ret_labels, ret_weights = [], [], [], [], []
    for snt_tokens, snt_starts, snt_ends, snt_outputs in zip(
        tokens, starts, ends, outputs
    ):
        ret_snt_starts, ret_snt_ends, ret_snt_labels, ret_snt_weights = [], [], [], []
        for start, end, output in zip(snt_starts, snt_ends, snt_outputs):
            if output.labels:
                ret_snt_starts.append(start)
                ret_snt_ends.append(end)
                ret_snt_labels.append(output.labels)
                ret_snt_labels.append(output.weights)
        ret_tokens.append(snt_tokens)
        ret_starts.append(ret_snt_starts)
        ret_ends.append(ret_snt_ends)
        ret_labels.append(ret_snt_labels)
        ret_weights.append(ret_snt_weights)

    features = get_msml_dataset_features(label_names, with_weitht=True)
    pseudo_dataset = Dataset.from_dict(
        {
            "tokens": ret_tokens,
            "starts": ret_starts,
            "ends": ret_ends,
            "labels": ret_labels,
            "weights": ret_weights,
        },
        info=DatasetInfo(description=json.dumps(desc), features=features),
    )
    return pseudo_dataset


def get_labels(ner_dataset: Dataset):
    return ner_dataset.features["labels"].feature.feature.names


import copy


def change_label_names(
    ner_dataset: Dataset, label_names: List[str], with_weight: bool = False
):
    info = copy.deepcopy(ner_dataset.info)
    old_names = info.features["labels"].feature.feature.names
    raw_label_names = []
    for snt in ner_dataset["labels"]:
        raw_label_names.append([[old_names[tag] for tag in span] for span in snt])
    desc = info.description
    new_ner_dataset = ner_dataset.to_dict()
    new_ner_dataset["labels"] = raw_label_names
    return Dataset.from_dict(
        new_ner_dataset,
        info=DatasetInfo(
            description=desc,
            features=get_msml_dataset_features(label_names, with_weight),
        ),
    )


def join_pseudo_and_gold_dataset(
    pseudo_dataset: Dataset, gold_dataset: DatasetDict
) -> DatasetDict:
    label_names = []
    label_names += pseudo_dataset.features["labels"].feature.feature.names
    label_names += gold_dataset["validation"].features["labels"].feature.feature.names
    label_names += gold_dataset["test"].features["labels"].feature.feature.names
    if "nc-O" in label_names:
        label_names = ["nc-O"] + sorted(set([l for l in label_names if l != "nc-O"]))
    else:
        label_names = sorted(set(label_names))
    ret = DatasetDict(
        {
            "train": change_label_names(pseudo_dataset, label_names, with_weight=True),
            "validation": change_label_names(gold_dataset["validation"], label_names),
            "test": change_label_names(gold_dataset["test"], label_names),
        }
    )
    return ret
