from hashlib import md5
from datasets import Dataset, DatasetDict
from pathlib import Path
from datasets.info import DatasetInfo
from seqeval.metrics.sequence_labeling import get_entities
import os
import datasets
from datasets import Dataset, DatasetDict
from src.utils.params import get_ner_dataset_features, task_name2ner_label_names
from src.ner_model.abstract_model import NERModel, NERModelConfig
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
from src.ner_model.two_stage import TwoStageConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig
from src.ner_model.typer.dict_match_typer import DictMatchTyperConfig
from omegaconf import MISSING

logger = getLogger(__name__)


@dataclass
class PseudoAnnoConfig:
    ner_model: NERModelConfig = TwoStageConfig(
        chunker=SpacyNPChunkerConfig(), typer=DictMatchTyperConfig()
    )
    output_dir: str = MISSING
    raw_corpus: str = MISSING
    gold_corpus: str = MISSING
    remove_fp_instance: bool = False
    add_erosion_fn: bool = False
    remove_misguidance_fn: bool = False
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
    raise NotImplementedError
    # TODO: 編集距離も追加する
    return new_tags


def add_erosion_entity(pred_tags, gold_tags):
    """Add erosion entity to the prediction tags.
    Erosion entity is a entity which is in gold tags but not in pred_tags.
    """
    raise NotImplementedError
    new_tags = ["O"] * len(pred_tags)
    for gold_label, s, e in get_entities(gold_tags):
        pass
    # TODO: 編集距離も追加する
    return new_tags


def remove_misguidance_fn(pred_tags, gold_tags):
    raise NotImplementedError
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
    # TODO: 編集距離も追加する
    return new_tags


def load_pseudo_dataset(
    raw_corpus: Dataset, ner_model: NERModel, conf: PseudoAnnoConfig
) -> Dataset:
    desc = dict()
    desc["raw_corpus"] = json.loads(raw_corpus.info.description)
    desc["ner_model"] = yaml.safe_load(OmegaConf.to_yaml(ner_model.conf))

    buffer_dir = Path(get_original_cwd()).joinpath(
        "data",
        "buffer",
        md5(("Pseudo Dataset from " + str(desc)).encode()).hexdigest(),
    )

    if not buffer_dir.exists():
        ret_tokens = []
        ner_tags = []
        if conf.remove_fp_instance or conf.add_erosion_fn or conf.remove_misguidance_fn:
            label_names = raw_corpus.features["ner_tags"].feature.names
            for tokens, gold_tags in tqdm(
                zip(raw_corpus["tokens"], raw_corpus["ner_tags"])
            ):
                pred_tags = ner_model.predict(tokens)
                gold_tags = [label_names[tagid] for tagid in gold_tags]
                if conf.remove_fp_instance:
                    pred_tags = remove_fp_ents(pred_tags, gold_tags)
                if conf.add_erosion_fn:
                    pred_tags = add_erosion_entity(pred_tags, gold_tags)
                if conf.remove_misguidance_fn:
                    pred_tags = remove_misguidance_fn(pred_tags, gold_tags)
                if any(tag != "O" for tag in pred_tags):
                    ret_tokens.append(tokens)
                    ner_tags.append(pred_tags)
        else:
            for tokens in tqdm(raw_corpus["tokens"]):
                pred_tags = ner_model.predict(tokens)
                if any(tag != "O" for tag in pred_tags):
                    ret_tokens.append(tokens)
                    ner_tags.append(pred_tags)

        ner_labels = [
            l
            for l, c in Counter([tag for snt in ner_tags for tag in snt]).most_common()
        ]
        pseudo_dataset = Dataset.from_dict(
            {"tokens": ret_tokens, "ner_tags": ner_tags},
            info=DatasetInfo(
                description=json.dumps(desc),
                features=get_ner_dataset_features(ner_labels),
            ),
        )
        pseudo_dataset.save_to_disk(str(buffer_dir))
    pseudo_dataset = Dataset.load_from_disk(str(buffer_dir))
    return pseudo_dataset


def get_tags(ner_dataset: Dataset):
    ner_labels = ner_dataset.features["ner_tags"].feature.names
    ner_tags = []
    for snt in ner_dataset["ner_tags"]:
        for tag in snt:
            ner_tags.append(ner_labels[tag])
    return ner_tags


import copy


def change_ner_label_names(ner_dataset: Dataset, label_names: List[str]):
    info = copy.deepcopy(ner_dataset.info)
    old_names = info.features["ner_tags"].feature.names
    raw_ner_tags = []
    for snt in ner_dataset["ner_tags"]:
        raw_ner_tags.append([old_names[tag] for tag in snt])
    desc = info.description
    new_ner_dataset = ner_dataset.to_dict()
    new_ner_dataset["ner_tags"] = raw_ner_tags
    return Dataset.from_dict(
        new_ner_dataset,
        info=DatasetInfo(
            description=desc, features=get_ner_dataset_features(label_names)
        ),
    )


def join_pseudo_and_gold_dataset(
    pseudo_dataset: Dataset, gold_dataset: DatasetDict
) -> DatasetDict:
    ner_tags = []
    ner_tags += get_tags(pseudo_dataset)
    ner_tags += get_tags(gold_dataset["validation"])
    ner_tags += get_tags(gold_dataset["test"])
    label_names = [l for l, c in Counter(ner_tags).most_common()]
    ret = DatasetDict(
        {
            "train": change_ner_label_names(pseudo_dataset, label_names),
            "validation": change_ner_label_names(
                gold_dataset["validation"], label_names
            ),
            "test": change_ner_label_names(gold_dataset["test"], label_names),
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
