from hashlib import md5
import os
from collections import defaultdict
from scipy.sparse import base
from seqeval.metrics.sequence_labeling import get_entities
from datasets import DatasetDict, Dataset
from src import ner_model
from src.ner_model.abstract_model import NERModel, NERModel
from pathlib import Path
from typing import List, Optional, Set
from copy import deepcopy
import logging
from seqeval.metrics.sequence_labeling import get_entities
from src.ner_model.chunker.abstract_model import Chunker
from src.ner_model.multi_label.ml_typer.abstract import MultiLabelTyper
from src.ner_model.two_stage import TwoStageModel
from src.utils.mlflow import MlflowWriter
from prettytable import PrettyTable
import statistics
from dataclasses import MISSING, dataclass
from src.ner_model.typer.abstract_model import TyperConfig
from hydra.core.config_store import ConfigStore
from src.ner_model.typer import typer_builder
from src.ner_model.typer.abstract_model import Typer
import numpy as np

logger = logging.getLogger(__name__)


def calculate_negative_precision(
    gold_ner_tags: List[List[str]], pred_ner_tags: List[List[str]]
):
    """Caluculate negative precision (whether predicted category doesn't partially match positive or not)"""
    negative_span_predicted_num = 0
    negative_span_has_no_duplication_num = 0
    for gold_tags, pred_tags in zip(gold_ner_tags, pred_ner_tags):
        for l, s, e in get_entities(pred_tags):
            if "nc-" in l:
                negative_span_predicted_num += 1
                if all(tag == "O" for tag in gold_tags[s : e + 1]):
                    negative_span_has_no_duplication_num += 1
    return negative_span_has_no_duplication_num / negative_span_predicted_num


def calculate_negative_token_PRF(
    gold_ner_tags: List[List[str]], pred_ner_tags: List[List[str]]
):
    """Calculate token level negative recall (how much negative tokens are predicted as negative)"""
    negative_token_gold = 0
    negative_token_predicted = 0
    negative_token_tp = 0  # i.e. tp for postive
    negative_cat_count = sum(["-nc-" in tag for tags in pred_ner_tags for tag in tags])
    for gold_tags, pred_tags in zip(gold_ner_tags, pred_ner_tags):
        for g_tag, p_tag in zip(gold_tags, pred_tags):
            if g_tag == "O":
                negative_token_gold += 1
            if "-nc-" in p_tag:
                negative_token_predicted += 1
            elif negative_cat_count == 0 and p_tag == "O":
                negative_token_predicted += 1

            if g_tag == "O" and "-nc-" in p_tag:
                negative_token_tp += 1
            elif negative_cat_count == 0 and g_tag == "O" and p_tag == "O":
                pass
    precision = negative_token_tp / negative_token_predicted
    recall = negative_token_tp / negative_token_gold
    if precision != 0 and recall != 0:
        f1 = 2 / (1 / precision + 1 / recall)
    else:
        f1 = 0
    return precision, recall, f1


def calculate_set_PRF(pred_set: Set, gold_set: Set):
    if len(pred_set) == 0:
        precision = 0
    else:
        precision = len(pred_set & gold_set) / len(pred_set)
    recall = len(pred_set & gold_set) / len(gold_set)
    if precision != 0 and recall != 0:
        f1 = 2 / (1 / precision + 1 / recall)
    else:
        f1 = 0
    return precision, recall, f1


@dataclass
class NERTestorConfig:
    baseline_typer: TyperConfig = MISSING


def register_ner_testor_configs(group="testor") -> None:
    cs = ConfigStore.instance()
    cs.store(group=group, name="base_NERTestor_config", node=NERTestorConfig)


class NERTestor:
    "NER Testor test NER Mondel on dataset"

    def __init__(
        self,
        ner_model: NERModel,
        ner_dataset: DatasetDict,
        writer: MlflowWriter,
        config: NERTestorConfig,
        chunker: Chunker = None,
    ) -> None:
        pass
        self.ner_model = ner_model
        self.datasets = ner_dataset
        self.datasets_hash = {
            key: split.__hash__() for key, split in ner_dataset.items()
        }
        self.args = {
            "ner_dataset": self.datasets_hash,
            "ner_model": ner_model.conf,
        }
        self.output_dir = Path(".")
        self.writer = writer
        logger.info("output_dir for NERTestor: %s" % str(self.output_dir))
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
        self.ner_tags = self.datasets["train"].features["ner_tags"].feature.names
        # self.labels = list(set(translate_ner_tags_into_classes(self.ner_tags)))

        from transformers.utils.logging import get_logger as transformer_get_logger

        trainer_logger = transformer_get_logger("transformers.trainer")
        import logging

        orig_level = trainer_logger.level
        trainer_logger.setLevel(logging.WARNING)

        # if isinstance(ner_model, TwoStageModel):
        #     self.baseline_typer = typer_builder(
        #         config.baseline_typer, ner_dataset, writer, chunker=None
        #     )
        #     self.analyze_likelihood_diff_between_dict_term(ner_dataset["test"])

        (
            self.prediction_for_test_w_nc,
            self.prediction_for_test,
        ) = self.load_prediction_for("test")
        (
            self.prediction_for_dev_w_nc,
            self.prediction_for_dev,
        ) = self.load_prediction_for("validation")

        trainer_logger.setLevel(orig_level)

        self.evaluate(self.prediction_for_test)
        self.evaluate_on_head(self.prediction_for_test)
        self.lenient_evaluate(self.prediction_for_test)
        self.analyze_fp(self.prediction_for_test)
        self.evaluate_by_sentence_length(self.prediction_for_test)
        self.evaluate_by_predicted_span_num(self.prediction_for_test)
        self.evaluate_span_detection()
        self.analyze_nc_fn(self.prediction_for_test_w_nc)
        self.evaluate_negative_category(self.prediction_for_test_w_nc)
        self.evaluate_negative_by_category(self.prediction_for_test_w_nc)
        if chunker:
            self.evaluate_on_negative_chunk(self.prediction_for_test_w_nc, chunker)
            self.evaluate_on_negative_chunk_by_category(
                self.prediction_for_test_w_nc, chunker
            )

    def analyze_likelihood_diff_between_dict_term(self, gold_dataset: Dataset):
        assert isinstance(self.ner_model, TwoStageModel)
        baseline_typer: Typer = self.baseline_typer
        ner_model: TwoStageModel = self.ner_model
        focus_typer: Typer = ner_model.typer
        in_dict_likelihoods = []
        out_dict_likelihoods = []
        label_names = focus_typer.label_names
        tag_names = gold_dataset.features["ner_tags"].feature.names
        tokens = gold_dataset["tokens"]
        starts, ends, labels = [], [], []
        for snt in gold_dataset:
            snt_starts = []
            snt_ends = []
            snt_labels = []
            for l, s, e in get_entities([tag_names[tag] for tag in snt["ner_tags"]]):
                snt_starts.append(s)
                snt_ends.append(e)
                snt_labels.append(l)
            starts.append(snt_starts)
            ends.append(snt_ends)
            labels.append(snt_labels)
        focus_probs = [
            output.probs for output in focus_typer.batch_predict(tokens, starts, ends)
        ]

        for snt_tokens, snt_starts, snt_ends, snt_labels, snt_probs in zip(
            tokens, starts, ends, labels, focus_probs
        ):
            baseline_predictions = baseline_typer.predict(
                snt_tokens, snt_starts, snt_ends
            ).labels
            for label, start, end, snt_prob, baseline_prediction in zip(
                snt_labels, snt_starts, snt_ends, snt_probs, baseline_predictions
            ):
                likelihood = snt_prob[label_names.index(label)]
                if isinstance(likelihood, np.float32):
                    likelihood = float(likelihood)
                if baseline_prediction == label:
                    in_dict_likelihoods.append(likelihood)
                    # self.modelのスパン (s, e)に対する予測確率を in_dict_likelihood に appendする
                    pass
                else:
                    out_dict_likelihoods.append(likelihood)
                    # self.modelのスパン (s, e)に対する予測確率を out_dict_likelihood に appendする
                    pass
        ptl = PrettyTable(["class", "mean", "variance"])
        in_dict_likelihood_mean = statistics.mean(in_dict_likelihoods)
        in_dict_likelihood_var = statistics.variance(in_dict_likelihoods)
        out_dict_likelihood_mean = statistics.mean(out_dict_likelihoods)
        out_dict_likelihood_var = statistics.variance(out_dict_likelihoods)
        self.writer.log_metric("in_dict_likelihood_mean", 100 * in_dict_likelihood_mean)
        self.writer.log_metric("in_dict_likelihood_var", 100 * in_dict_likelihood_var)
        self.writer.log_metric(
            "out_dict_likelihood_mean", 100 * out_dict_likelihood_mean
        )
        self.writer.log_metric("out_dict_likelihood_var", 100 * out_dict_likelihood_var)
        ptl.add_row(
            ["in dict", in_dict_likelihood_mean, in_dict_likelihood_var],
        )
        ptl.add_row(
            [
                "out dict",
                statistics.mean(out_dict_likelihoods),
                statistics.variance(out_dict_likelihoods),
            ],
        )

        logger.info(ptl.get_string())

    def analyze_nc_fn(self, prediction_for_test_w_nc: Dataset):
        count_for_fn_miss_classification_on_end = 0
        count_for_fn_miss_classification_on_non_end = 0
        for snt in prediction_for_test_w_nc:
            pred_tags = snt["pred_ner_tags"]
            gold_tags = snt["gold_ner_tags"]
            for l, s, e in get_entities(pred_tags):
                if l.startswith("nc-"):
                    gold_labels = [l for l, s, e in get_entities(gold_tags[s : e + 1])]
                    gold_ends = [
                        s + e2 for l, s2, e2 in get_entities(gold_tags[s : e + 1])
                    ]
                    if gold_labels:  # check fn or not
                        if e in gold_ends:
                            count_for_fn_miss_classification_on_end += 1
                        else:
                            count_for_fn_miss_classification_on_non_end += 1
        ptl = PrettyTable(["class", "count", "ratio (%)"])
        fp_num = (
            +count_for_fn_miss_classification_on_end
            + count_for_fn_miss_classification_on_non_end
        )
        if fp_num == 0:
            ptl.add_row(
                [
                    "miss nc on end",
                    0,
                    0,
                ],
            )
            ptl.add_row(
                [
                    "miss classification on non-end",
                    0,
                    0,
                ],
            )
        else:
            ptl.add_row(
                [
                    "miss nc on end",
                    count_for_fn_miss_classification_on_end,
                    count_for_fn_miss_classification_on_end / fp_num * 100,
                ],
            )
            ptl.add_row(
                [
                    "miss classification on non-end",
                    count_for_fn_miss_classification_on_non_end,
                    count_for_fn_miss_classification_on_non_end / fp_num * 100,
                ],
            )
        logger.info(ptl.get_string())

    def get_np_negative_chunks(self, prediction_w_nc: Dataset, chunker: Chunker):
        # evaluate on NP
        np_negative_chunks = set()
        all_np_chunks = set()
        for sid, (gold, snt) in enumerate(
            zip(prediction_w_nc["gold_ner_tags"], prediction_w_nc["tokens"])
        ):
            chunks = chunker.predict(snt)
            for s, e in chunks:
                all_np_chunks.add((sid, s, e))
                if all(tag == "O" for tag in gold[s:e]):
                    np_negative_chunks.add((sid, s, e))
        return all_np_chunks, np_negative_chunks
        pass

    def analyze_fp(self, prediction_for_test: Dataset):
        """
        Analyze the false positive (FP) in the prediction.
        """
        # Check whether the error on "O" or another category
        count_for_fp_on_o = 0
        count_for_fp_miss_classification_on_end = 0
        count_for_fp_miss_classification_on_non_end = 0
        for snt in prediction_for_test:
            pred_tags = snt["pred_ner_tags"]
            gold_tags = snt["gold_ner_tags"]
            for l, s, e in get_entities(pred_tags):
                gold_labels = [l for l, s, e in get_entities(gold_tags[s : e + 1])]
                gold_ends = [e for l, s, e in get_entities(gold_tags[s : e + 1])]
                if l in gold_labels:  # check fp or not
                    if e in gold_ends:
                        count_for_fp_miss_classification_on_end += 1
                    else:
                        count_for_fp_miss_classification_on_non_end += 1
                elif all(tag == "O" for tag in gold_tags[s : e + 1]):
                    count_for_fp_on_o += 1
        ptl = PrettyTable(["class", "count", "ratio (%)"])
        fp_num = (
            count_for_fp_on_o
            + count_for_fp_miss_classification_on_end
            + count_for_fp_miss_classification_on_non_end
        )
        if fp_num == 0:
            ptl.add_row(["on all O", count_for_fp_on_o, 0])
            ptl.add_row(
                [
                    "miss classification on end",
                    count_for_fp_miss_classification_on_end,
                    0,
                ],
            )
            ptl.add_row(
                [
                    "miss classification on non-end",
                    count_for_fp_miss_classification_on_non_end,
                    0,
                ],
            )
        else:
            ptl.add_row(
                ["on all O", count_for_fp_on_o, count_for_fp_on_o / fp_num * 100]
            )
            ptl.add_row(
                [
                    "miss classification on end",
                    count_for_fp_miss_classification_on_end,
                    count_for_fp_miss_classification_on_end / fp_num * 100,
                ],
            )
            ptl.add_row(
                [
                    "miss classification on non-end",
                    count_for_fp_miss_classification_on_non_end,
                    count_for_fp_miss_classification_on_non_end / fp_num * 100,
                ],
            )
        logger.info(ptl.get_string())

    def get_np_negative_chunks(self, prediction_w_nc: Dataset, chunker: Chunker):
        # evaluate on NP
        np_negative_chunks = set()
        all_np_chunks = set()
        for sid, (gold, snt) in enumerate(
            zip(prediction_w_nc["gold_ner_tags"], prediction_w_nc["tokens"])
        ):
            chunks = chunker.predict(snt)
            for s, e in chunks:
                all_np_chunks.add((sid, s, e))
                if all(tag == "O" for tag in gold[s:e]):
                    np_negative_chunks.add((sid, s, e))
        return all_np_chunks, np_negative_chunks

    def get_enumerated_negative_spans(self, prediction_w_nc: Dataset, chunker: Chunker):
        # on enuemerated f1
        from src.utils.params import span_length

        ## (1) enumerate candidate spans
        candidate_spans = set()
        for sid, snt in enumerate(prediction_w_nc):
            candidate_span = set(
                [
                    (sid, i, j)
                    for i in range(len(snt["tokens"]))
                    for j in range(i, len(snt["tokens"]))
                    if j - i <= span_length
                ]
            )
            candidate_spans |= candidate_span
        ## (2) delete gold spans from (1)
        gold_chunks = set(
            [
                (sid, s, e + 1)
                for sid, gold in enumerate(prediction_w_nc["gold_ner_tags"])
                for l, s, e in get_entities(gold)
            ]
        )
        enumerated_negative_spans = candidate_spans - gold_chunks
        return candidate_spans, enumerated_negative_spans

    def evaluate_on_negative_chunk(self, prediction_w_nc: Dataset, chunker: Chunker):
        all_np_chunks, np_negative_chunks = self.get_np_negative_chunks(
            prediction_w_nc, chunker
        )
        candidate_spans, enumerated_negative_spans = self.get_enumerated_negative_spans(
            prediction_w_nc, chunker
        )
        ## evaluate strict P./R./F.
        pred_negative_chunks = set(
            [
                (sid, s, e + 1)
                for sid, pred in enumerate(prediction_w_nc["pred_ner_tags"])
                for l, s, e in get_entities(pred)
                if "nc-" in l
            ]
        )
        if not pred_negative_chunks:
            pred_positive_chunks = set(
                [
                    (sid, s, e + 1)
                    for sid, pred in enumerate(prediction_w_nc["pred_ner_tags"])
                    for l, s, e in get_entities(pred)
                    if not l.startswith("nc-")
                ]
            )
            pred_negative_np = all_np_chunks - pred_positive_chunks
            pred_negative_enumerated = candidate_spans - pred_positive_chunks
        else:
            pred_negative_np = pred_negative_chunks
            pred_negative_enumerated = pred_negative_chunks

        precision, recall, f1 = calculate_set_PRF(pred_negative_np, np_negative_chunks)
        logger.info(
            "P./R./F. for NP Negatives (%%) : %.2f | %.2f | %.2f "
            % (100 * precision, 100 * recall, 100 * f1)
        )

        precision, recall, f1 = calculate_set_PRF(
            pred_negative_enumerated, enumerated_negative_spans
        )
        # logging result
        logger.info(
            "P./R./F. for enumerated negatives (%%) : %.2f | %.2f | %.2f "
            % (100 * precision, 100 * recall, 100 * f1)
        )

    def evaluate_on_negative_chunk_by_category(
        self, prediction_w_nc: Dataset, chunker: Chunker
    ):
        all_np_chunks, np_negative_chunks = self.get_np_negative_chunks(
            prediction_w_nc, chunker
        )
        candidate_spans, enumerated_negative_spans = self.get_enumerated_negative_spans(
            prediction_w_nc, chunker
        )

        # 何周もするの大変なので、一周だけするようにする
        nc2pred_negative_chunks = defaultdict(set)
        for sid, pred in enumerate(prediction_w_nc["pred_ner_tags"]):
            for l, s, e in get_entities(pred):
                if l.startswith("nc-"):
                    nc2pred_negative_chunks[l].add((sid, s, e + 1))
        # TODO: Negative Categoryがないときの評価を追加する

        ncs = []
        np_negative_prf = []
        enumerated_negative_prf = []
        for nc, pred_negative_chunks in nc2pred_negative_chunks.items():
            ncs.append(nc[3:])
            precision, recall, f1 = calculate_set_PRF(
                pred_negative_chunks, np_negative_chunks
            )
            np_negative_prf.append((precision, recall, f1))

            precision, recall, f1 = calculate_set_PRF(
                pred_negative_chunks, enumerated_negative_spans
            )
            enumerated_negative_prf.append((precision, recall, f1))

        ptl = PrettyTable()
        ptl.add_column("# negative category", ncs)
        ptl.add_column(
            "negative np precision (%)",
            ["%.2f" % (100 * p,) for p, r, f in np_negative_prf],
        )
        ptl.add_column(
            "negative np recall (%)",
            ["%.2f" % (100 * r,) for p, r, f in np_negative_prf],
        )
        ptl.add_column(
            "negative np f1 (%)", ["%.2f" % (100 * f,) for p, r, f in np_negative_prf]
        )
        ptl.add_column(
            "negative enumerated precision (%)",
            ["%.2f" % (100 * p,) for p, r, f in enumerated_negative_prf],
        )
        ptl.add_column(
            "negative enumerated recall (%)",
            ["%.2f" % (100 * r,) for p, r, f in enumerated_negative_prf],
        )
        ptl.add_column(
            "negative enumerated f1 (%)",
            ["%.2f" % (100 * f,) for p, r, f in enumerated_negative_prf],
        )
        logger.info(ptl.get_string())

    def load_prediction_for(self, split="test"):
        # transformersのlogging level変更

        # 変数初期化
        tokens = []
        gold_ner_tags = []
        ner_tag_names = self.datasets[split].features["ner_tags"].feature.names
        pred_ner_tags = []
        logger.info("Start predictions for %s." % split)
        tokens = self.datasets[split]["tokens"]
        # _dict_ner_tags = self.dict_match_ner_model.batch_predict(tokens)
        _pred_ner_tags = self.ner_model.batch_predict(tokens)
        # pred_ner_tagsから fake_cat_で始まるラベルをのぞく
        _gold_ner_tags = [
            [ner_tag_names[tag] for tag in snt["ner_tags"]]
            for snt in self.datasets[split]
        ]
        tokens = []
        pred_ner_tags = []
        # dict_ner_tags = []
        gold_ner_tags = []
        for doc, _pred, _gold in zip(
            self.datasets[split],
            _pred_ner_tags,
            _gold_ner_tags,
            #  _dict_ner_tags
        ):
            tokens.append(doc["tokens"])
            pred_ner_tags.append(_pred)
            gold_ner_tags.append(_gold)
            # dict_ner_tags.append(_dict)
        assert len(tokens) == len(gold_ner_tags)
        assert len(gold_ner_tags) == len(pred_ner_tags)
        # assert len(gold_ner_tags) == len(dict_ner_tags)
        with open(
            os.path.join(self.output_dir, "prediction_for_%s.txt" % split), "w"
        ) as f:
            # f.write("token\tdictmatch\tpred\tgold\n")
            f.write("token\tpred\tgold\n")
            f.write(
                "\n\n".join(
                    [
                        "\n".join(
                            [
                                "\t".join(w)
                                for w in zip(
                                    tok,
                                    # dnt,
                                    pnt,
                                    gnt,
                                )
                            ]
                        )
                        for tok, pnt, gnt in zip(
                            tokens,
                            # dict_ner_tags,
                            pred_ner_tags,
                            gold_ner_tags,
                        )
                    ]
                )
            )
        pred_ner_tags_wo_nc = [
            ["O" if "-nc-" in tag else tag for tag in tags] for tags in pred_ner_tags
        ]
        prediction_for_split = Dataset.from_dict(
            {
                "tokens": tokens,
                "gold_ner_tags": gold_ner_tags,
                "pred_ner_tags": pred_ner_tags,
                # "dict_ner_tags": dict_ner_tags,
            }
        )
        prediction_for_split.save_to_disk(
            os.path.join(self.output_dir, "prediction_for_%s.dataset" % split)
        )

        prediction_for_split_wo_nc = Dataset.from_dict(
            {
                "tokens": tokens,
                "gold_ner_tags": gold_ner_tags,
                "pred_ner_tags": pred_ner_tags_wo_nc,
                # "dict_ner_tags": dict_ner_tags,
            }
        )
        return prediction_for_split, prediction_for_split_wo_nc

    def evaluate_negative_category(self, prediction_w_nc: Dataset):
        (
            negative_token_precision,
            negative_token_recall,
            _,
        ) = calculate_negative_token_PRF(
            prediction_w_nc["gold_ner_tags"], prediction_w_nc["pred_ner_tags"]
        )
        logger.info(
            "negative token precision: %.2f" % (100 * negative_token_precision,)
        )
        logger.info("negative token recall: %.2f" % (100 * negative_token_recall,))
        return negative_token_precision, negative_token_recall

    def evaluate_negative_by_category(self, prediction_w_nc: Dataset):
        gold_ner_tags = prediction_w_nc["gold_ner_tags"]
        pred_ner_tags = prediction_w_nc["pred_ner_tags"]
        ncs = sorted(
            list(
                set(
                    [
                        l
                        for l, s, e in get_entities(pred_ner_tags)
                        if l.startswith("nc-")
                    ]
                )
            )
        )
        negative_token_precisions = []
        negative_token_recalls = []
        for nc in ncs:
            pred_ner_tags_w_only_focus_nc = []
            for snt in pred_ner_tags:
                replaced_tags = []
                for tag in snt:
                    if "-nc-" in tag:
                        if tag[2:] == nc:
                            replaced_tags.append(tag)
                        else:
                            replaced_tags.append("O")
                    else:
                        replaced_tags.append(tag)
                pred_ner_tags_w_only_focus_nc.append(replaced_tags)
            precision, recall, f1 = calculate_negative_token_PRF(
                gold_ner_tags, pred_ner_tags_w_only_focus_nc
            )
            negative_token_precisions.append("%.2f" % (100 * precision,))
            negative_token_recalls.append("%.2f" % (100 * recall,))
        from prettytable import PrettyTable

        ptl = PrettyTable()
        ptl.add_column("# negative category", ncs)
        ptl.add_column("negative token precision (%)", negative_token_precisions)
        ptl.add_column("negative token recall (%)", negative_token_recalls)
        logger.info(ptl.get_string())

    def add_dict_match_prediction(self, pred_dataset):
        raise NotImplementedError

    def evaluate(self, prediction_for_test: Dataset):
        logger.info("evaluate strict ner score")
        from seqeval.metrics import (
            classification_report,
            precision_score,
            recall_score,
            f1_score,
        )

        logger.info(
            classification_report(
                prediction_for_test["gold_ner_tags"],
                prediction_for_test["pred_ner_tags"],
                digits=4,
            )
        )
        P = precision_score(
            prediction_for_test["gold_ner_tags"], prediction_for_test["pred_ner_tags"]
        )
        R = recall_score(
            prediction_for_test["gold_ner_tags"], prediction_for_test["pred_ner_tags"]
        )
        F = f1_score(
            prediction_for_test["gold_ner_tags"], prediction_for_test["pred_ner_tags"]
        )
        logger.info("| P. | R. | F. |")
        logger.info("| %.2f | %.2f | %.2f |" % (100 * P, 100 * R, 100 * F))
        self.writer.log_metric("precision", 100 * P)
        self.writer.log_metric("recall", 100 * R)
        self.writer.log_metric("f1", 100 * F)

    def evaluate_on_head(self, prediction_for_test: Dataset):
        logger.info("evaluate on head")
        gold = set()
        pred = set()
        for sid, snt in enumerate(prediction_for_test):
            gold |= {(sid, e, l) for l, s, e in get_entities(snt["gold_ner_tags"])}
            pred |= {(sid, e, l) for l, s, e in get_entities(snt["pred_ner_tags"])}
        tp = len(gold & pred)
        if tp != 0:
            precision = tp / len(pred)
            recall = tp / len(gold)
            f1 = 2 / (1 / precision + 1 / recall)
        else:
            precision, recall, f1 = 0, 0, 0
        logger.info("P/R/F=%.4f/%.4f/%.4f" % (precision, recall, f1))
        slots = list(set([(sid, e) for sid, e, l in gold | pred]))
        from collections import defaultdict

        slot2gold = defaultdict(lambda: "O")
        slot2gold.update(({(sid, e): l for sid, e, l in gold}))
        slot2pred = defaultdict(lambda: "O")
        slot2pred.update(({(sid, e): l for sid, e, l in pred}))
        _gold = [slot2gold[(sid, e)] for sid, e in slots]
        _pred = [slot2pred[(sid, e)] for sid, e in slots]
        from sklearn.metrics import classification_report

        logger.info(classification_report(_gold, _pred))

    def lenient_evaluate(self, prediction_for_test: Dataset):
        logger.info("evaluate lenient ner score")
        from seqeval.metrics import classification_report

        pred_tps = defaultdict(lambda: 0)
        gold_tps = defaultdict(lambda: 0)
        pred_nums = defaultdict(lambda: 0)
        gold_nums = defaultdict(lambda: 0)
        for gold_tag, pred_tag in zip(
            prediction_for_test["gold_ner_tags"], prediction_for_test["pred_ner_tags"]
        ):
            for pl, ps, pe in get_entities(pred_tag):
                pred_span = set(range(ps, pe + 1))
                pred_nums[pl] += 1
                for gl, gs, ge in get_entities(gold_tag):
                    gold_span = set(range(gs, ge + 1))
                    if pred_span & gold_span and pl == gl:
                        pred_tps[pl] += 1
                        break

            for gl, gs, ge in get_entities(gold_tag):
                gold_span = set(range(gs, ge + 1))
                gold_nums[gl] += 1
                for pl, ps, pe in get_entities(pred_tag):
                    pred_span = set(range(ps, pe + 1))
                    if pred_span & gold_span and pl == gl:
                        gold_tps[pl] += 1
                        break
                    pass
        precisions = dict()
        recalls = dict()
        f1s = dict()
        logger.info("| class | P. | R. | F. |")
        logger.info("|----------------------|")
        for l in pred_tps:
            precisions[l] = pred_tps[l] / pred_nums[l]
            recalls[l] = gold_tps[l] / gold_nums[l]
            f1s[l] = 2 / (1 / precisions[l] + 1 / recalls[l])
            logger.info(
                "| %s | %.2f | %.2f | %.2f |"
                % (l, 100 * precisions[l], 100 * recalls[l], 100 * f1s[l])
            )
        if sum(pred_nums.values()) == 0:
            precision = 0
        else:
            precision = sum(pred_tps.values()) / sum(pred_nums.values())
        recall = sum(gold_tps.values()) / sum(gold_nums.values())
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 / (1 / precision + 1 / recall)
        logger.info(
            "| total | %.2f | %.2f | %.2f |" % (100 * precision, 100 * recall, 100 * f1)
        )
        self.writer.log_metric("lenient P.", 100 * precision)
        self.writer.log_metric("lenient R.", 100 * recall)
        self.writer.log_metric("lenient F.", 100 * f1)
        # logger.info(
        #     classification_report(
        #         prediction_for_test["gold_ner_tags"],
        #         prediction_for_test["pred_ner_tags"],
        #     )
        # )

    def evaluate_by_sentence_length(self, predictions: Dataset):
        bin_splits = list(
            range(0, max([len(snt["tokens"]) for snt in predictions]) + 5, 5)
        )
        from seqeval.metrics import (
            classification_report,
            precision_score,
            recall_score,
            f1_score,
        )

        bins = list(zip(bin_splits, bin_splits[1:]))
        bin2snts = defaultdict(list)

        for snt in predictions:
            for (s, e) in bins:
                if s < len(snt["tokens"]) and len(snt["tokens"]) <= e:
                    bin2snts[(s, e)].append(snt)

        precisions = []
        recalls = []
        f1s = []

        for bin, snts in bin2snts.items():
            pred_ner_tags, gold_ner_tags = [], []
            for snt in snts:
                pred_ner_tags.append(snt["pred_ner_tags"])
                gold_ner_tags.append(snt["gold_ner_tags"])
            precisions.append(precision_score(gold_ner_tags, pred_ner_tags))
            recalls.append(recall_score(gold_ner_tags, pred_ner_tags))
            f1s.append(f1_score(gold_ner_tags, pred_ner_tags))
            pass

        logger.info("evaluate by sentence length: ")
        logger.info("|   |" + " | ".join("(%d, %d]" % (s, e) for s, e in bins) + " |")
        logger.info("|-" + "|-" * len(bins) + "|-|")
        logger.info(
            "| precision |" + " | ".join(["%.2f" % (100 * p,) for p in precisions])
        )
        logger.info(
            "| recall    |" + " | ".join(["%.2f" % (100 * r,) for r in recalls])
        )
        logger.info("| f1        |" + " | ".join(["%.2f" % (100 * f,) for f in f1s]))

    def evaluate_by_predicted_span_num(self, prediction_for_test: Dataset):
        logger.info("evaluate by predicted span num per snt")
        pass
        pred_num2snts = defaultdict(list)
        for snt in prediction_for_test:
            pred_num2snts[len(get_entities(snt["pred_ner_tags"]))].append(snt)
        pred_num2prec = defaultdict(list)
        pred_num2rec = defaultdict(list)
        pred_num2f1 = defaultdict(list)
        from seqeval.metrics import precision_score, recall_score, f1_score

        for k, snts in pred_num2snts.items():
            pred_tags = [snt["pred_ner_tags"] for snt in snts]
            gold_tags = [snt["gold_ner_tags"] for snt in snts]
            pred_num2prec[k] = precision_score(gold_tags, pred_tags)
            pred_num2rec[k] = recall_score(gold_tags, pred_tags)
            pred_num2f1[k] = f1_score(gold_tags, pred_tags)
        pass
        from prettytable import PrettyTable

        ptl = PrettyTable()
        pred_nums = sorted(pred_num2snts.keys())
        ptl.field_names = ["# pred/snt"] + sorted(pred_num2snts.keys())
        ptl.add_row(["# snt"] + [len(pred_num2snts[n]) for n in pred_nums])
        ptl.add_row(
            ["precision"] + ["{:.2f}".format(100 * pred_num2prec[n]) for n in pred_nums]
        )
        ptl.add_row(
            ["recall"] + ["{:.2f}".format(100 * pred_num2rec[n]) for n in pred_nums]
        )
        ptl.add_row(["f1"] + ["{:.2f}".format(100 * pred_num2f1[n]) for n in pred_nums])
        logger.info(ptl.get_string())

    def evaluate_span_detection(self):
        logger.info("evaluate span detection")
        gold = []
        pred = []
        for snt in self.prediction_for_test:
            gold.append(
                [
                    tag.replace(tag[2:], "span") if tag != "O" else "O"
                    for tag in snt["gold_ner_tags"]
                ]
            )
            pred.append(
                [
                    tag.replace(tag[2:], "span") if tag != "O" else "O"
                    for tag in snt["pred_ner_tags"]
                ]
            )
        from seqeval.metrics import classification_report

        logger.info(classification_report(gold, pred))

    def evaluate_difference_from_dict(self):
        logger.info("evaluate on difference from dict")
        false_negative_gold_ner_tags = []
        false_negative_pred_ner_tags = []

        for snt in self.prediction_for_test:
            true_positives = set(get_entities(snt["gold_ner_tags"])) & set(
                get_entities(snt["dict_ner_tags"])
            )
            _missed_gold_tags = deepcopy(snt["gold_ner_tags"])
            _fp_pred_tags = deepcopy(snt["pred_ner_tags"])
            for l, s, e in true_positives:
                for tokid in range(s, e + 1):
                    _missed_gold_tags[tokid] = "O"
                    _fp_pred_tags[tokid] = "O"
            false_negative_gold_ner_tags.append(_missed_gold_tags)
            false_negative_pred_ner_tags.append(_fp_pred_tags)
        from seqeval.metrics import classification_report

        logger.info(
            classification_report(
                false_negative_gold_ner_tags,
                false_negative_pred_ner_tags,
            )
        )

    def evaluate_false_negative_from_dict_on_head(self):
        logger.info("Start evaluate false negative examples on head")
        pred = set()
        gold = set()
        for sid, snt in enumerate(self.prediction_for_test):
            _pred = get_entities(snt["pred_ner_tags"])
            _dict = get_entities(snt["dict_ner_tags"])
            _gold = get_entities(snt["gold_ner_tags"])
            dict_spans = [set(range(s, e + 1)) for l, s, e in _dict]
            _pred = [
                (sid, e, l)
                for l, s, e in _pred
                if not any(set(range(s, e + 1)) & span for span in dict_spans)
            ]
            pred |= set(_pred)
            _gold = [
                (sid, e, l)
                for l, s, e in _gold
                if not any(set(range(s, e + 1)) & span for span in dict_spans)
            ]
            gold |= set(_gold)

        tp = len(pred & gold)
        if tp != 0:
            precision = tp / len(pred)
            recall = tp / len(gold)
            f1 = 2 / (1 / precision + 1 / recall)
        else:
            precision, recall, f1 = 0, 0, 0
        logger.info("P/R/F=%.2f|%.2f|%.2f" % (precision, recall, f1))

        slots = list(set([(sid, e) for sid, e, l in gold | pred]))
        from collections import defaultdict

        slot2gold = defaultdict(lambda: "O")
        slot2gold.update(({(sid, e): l for sid, e, l in gold}))
        slot2pred = defaultdict(lambda: "O")
        slot2pred.update(({(sid, e): l for sid, e, l in pred}))
        _gold = [slot2gold[(sid, e)] for sid, e in slots]
        _pred = [slot2pred[(sid, e)] for sid, e in slots]
        from sklearn.metrics import classification_report

        logger.info(classification_report(_gold, _pred))

    def evaluate_on_span_mismatch(self):
        logger.info("evaluate on span_mismatch")
        gold = []
        for snt in self.prediction_for_test:
            gold_tags = deepcopy(snt["gold_ner_tags"])
            dict_spans = [
                (s, e + 1, l) for l, s, e in get_entities(snt["dict_ner_tags"])
            ]
            gold_spans = [
                (s, e + 1, l) for l, s, e in get_entities(snt["gold_ner_tags"])
            ]
            for gs, ge, gl in gold_spans:
                for ds, de, dl in dict_spans:
                    if (
                        set(range(gs, ge)) & set(range(ds, de))
                        and gl == dl
                        and (gs, ge) != (ds, de)
                    ):
                        break
                else:
                    for tokid in range(gs, ge):
                        gold_tags[tokid] = "O"
            gold.append(gold_tags)

        from seqeval.metrics import classification_report

        logger.info(
            classification_report(gold, self.prediction_for_test["pred_ner_tags"])
        )
        pass

    def evaluate_on_label_mismatch(self):
        logger.info("evaluate on label mismatch")
        gold = []
        for snt in self.prediction_for_test:
            gold_tags = deepcopy(snt["gold_ner_tags"])
            dict_spans = [
                (s, e + 1, l) for l, s, e in get_entities(snt["dict_ner_tags"])
            ]
            gold_spans = [
                (s, e + 1, l) for l, s, e in get_entities(snt["gold_ner_tags"])
            ]
            for gs, ge, gl in gold_spans:
                for ds, de, dl in dict_spans:
                    if (gs, ge) == (ds, de) and gl != dl:
                        break
                else:
                    for tokid in range(gs, ge):
                        gold_tags[tokid] = "O"
            gold.append(gold_tags)
        from seqeval.metrics import classification_report

        logger.info(
            classification_report(gold, self.prediction_for_test["pred_ner_tags"])
        )
