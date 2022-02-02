from hashlib import md5
from datasets import Dataset, DatasetDict
from pathlib import Path
from datasets.info import DatasetInfo
from seqeval.metrics.sequence_labeling import get_entities
import os
import datasets
from datasets import Dataset, DatasetDict
from src.utils.params import get_ner_dataset_features, task_name2ner_label_names
from src.ner_model.multi_label.abstract_model import (
    MultiLabelNERModel,
    MultiLabelNERModelConfig,
)
from hashlib import md5
from src.utils.params import pseudo_annotated_time
import shutil
from dataclasses import dataclass
from logging import getLogger
from tqdm import tqdm
from collections import Counter
import json
import yaml
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
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
    # remove_fp_instance: bool = False
    # mark_misguided_fn: bool = False
    # duplicate_cats: str = MISSING
    # focus_cats: str = MISSING


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


def mark_misguided_fn(pred_tags: List[str], gold_tags: List[str]):
    new_tags = copy.deepcopy(pred_tags)
    for gold_label, gs, ge in get_entities(gold_tags):
        pred_labels = {
            pl
            for pl, ps, pe in get_entities(pred_tags[gs : ge + 1])
            if not pl.startswith("nc")
        }
        if not gold_label in pred_labels:
            for i in range(gs, ge + 1):
                if i == gs:
                    new_tags[i] = "B-MISGUIDANCE"
                else:
                    new_tags[i] = "I-MISGUIDANCE"
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


def get_msml_dataset_features(label_names):
    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "starts": datasets.Sequence(datasets.Value("int32")),
            "ends": datasets.Sequence(datasets.Value("int32")),
            "labels": datasets.Sequence(
                datasets.Sequence(datasets.ClassLabel(names=label_names))
            ),
        }
    )
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
    ret_tokens, ret_starts, ret_ends, ret_labels = [], [], [], []
    for snt_tokens, snt_starts, snt_ends, snt_outputs in zip(
        tokens, starts, ends, outputs
    ):
        if snt_outputs:
            ret_tokens.append(snt_tokens)
            ret_starts.append(snt_starts)
            ret_ends.append(snt_ends)
            ret_labels.append(snt_outputs)

    # multi_label_ner_model
    # for tokens in tqdm(raw_corpus["tokens"]):
    #     pred_tags = multi_label_ner_model.predict(tokens)
    #     if any(tag != "O" for tag in pred_tags):
    #         ret_tokens.append(tokens)
    #         ner_tags.append(pred_tags)

    # ner_labels = [
    #     l for l, c in Counter([tag for snt in ner_tags for tag in snt]).most_common()
    # ]
    features = get_msml_dataset_features(label_names)
    pseudo_dataset = Dataset.from_dict(
        {
            "tokens": ret_tokens,
            "starts": ret_starts,
            "ends": ret_ends,
            "labels": ret_labels,
        },
        info=DatasetInfo(description=json.dumps(desc), features=features),
    )
    return pseudo_dataset


def get_labels(ner_dataset: Dataset):
    return ner_dataset.features["labels"].feature.feature.names


import copy


def change_label_names(ner_dataset: Dataset, label_names: List[str]):
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
            description=desc, features=get_msml_dataset_features(label_names)
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
            "train": change_label_names(pseudo_dataset, label_names),
            "validation": change_label_names(gold_dataset["validation"], label_names),
            "test": change_label_names(gold_dataset["test"], label_names),
        }
    )
    return ret


# class PseudoDataset:
#     def __init__(
#         self, raw_corpus: RawCorpusDataset, ner_model: NERModel, task: str
#     ) -> None:
#         self.ner_label_names = ner_model.label_names
#         self.unlabelled_corpus = raw_corpus.load_tokens()
#         self.ner_model = ner_model
#         pass
#         self.args = {
#             "raw_corpus": raw_corpus.args,
#             "ner_model": ner_model.args,
#             "task": task,
#         }
#         self.output_dir = Path("data/buffer").joinpath(
#             md5(str(self.args).encode()).hexdigest()
#         )
#         if (
#             self.output_dir.exists()
#             and self.output_dir.stat().st_ctime < pseudo_annotated_time
#         ):
#             # dict_balanced corpusで見つけられた単語に対して事例を収集するように改変　そのためそれ以降に作成されたデータでないときには再度データを作り直す
#             shutil.rmtree(self.output_dir)
#             logger.info("file is old, so it is deleted")
#         logger.info("pseudo annotation output dir: %s" % str(self.output_dir))
#         self.datasets = self.load_splitted_dict_matched()

#     def load_dictionary_matched_text(self) -> Dataset:
#         dict_matched_dir = Path(os.path.join(self.output_dir, "dict_matched"))
#         if not dict_matched_dir.exists():
#             pre_ner_tagss = self.ner_model.batch_predict(
#                 self.unlabelled_corpus["tokens"], self.unlabelled_corpus["POS"]
#             )
#             tokenss, ner_tagss, boss, poss = [], [], [], []
#             for tok, bos, pos, nt in zip(
#                 self.unlabelled_corpus["tokens"],
#                 self.unlabelled_corpus["bos_ids"],
#                 self.unlabelled_corpus["POS"],
#                 pre_ner_tagss,
#             ):
#                 if get_entities(nt):
#                     tokenss.append(tok)
#                     ner_tagss.append([self.ner_label_names.index(tag) for tag in nt])
#                     boss.append(bos)
#                     poss.append(pos)
#             info = datasets.DatasetInfo(
#                 features=get_ner_dataset_features(self.ner_label_names)
#             )
#             doc_id = list(range(len(tokenss)))
#             snt_id = [-1] * len(tokenss)
#             dict_matched = Dataset.from_dict(
#                 {
#                     "tokens": tokenss,
#                     "ner_tags": ner_tagss,
#                     "doc_id": doc_id,
#                     "snt_id": snt_id,
#                     "bos_ids": boss,
#                     "POS": poss,
#                 },
#                 info=info,
#             )
#             dict_matched.save_to_disk(dict_matched_dir)
#         dict_matched = Dataset.load_from_disk(dict_matched_dir)
#         return dict_matched

#     def load_splitted_dict_matched(self) -> DatasetDict:
#         splitted_dict_matched_dir = Path(
#             os.path.join(self.output_dir, "splitted_dict_matched")
#         )
#         if not splitted_dict_matched_dir.exists():
#             self.labeled_snts = self.load_dictionary_matched_text()
#             train_validation_split = int(0.9 * len(self.labeled_snts))
#             train, validation = (
#                 self.labeled_snts[:train_validation_split],
#                 self.labeled_snts[train_validation_split:],
#             )
#             train = Dataset.from_dict(train, info=self.labeled_snts.info)
#             validation = Dataset.from_dict(validation, info=self.labeled_snts.info)
#             splitted_dict_matched = DatasetDict(
#                 {"train": train, "validation": validation}
#             )
#             splitted_dict_matched.save_to_disk(splitted_dict_matched_dir)
#         splitted_dict_matched = DatasetDict.load_from_disk(splitted_dict_matched_dir)
#         return splitted_dict_matched
