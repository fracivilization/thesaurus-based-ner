from hashlib import md5
import os
from collections import defaultdict
from seqeval.metrics.sequence_labeling import get_entities
from datasets import DatasetDict, Dataset
from src.ner_model.abstract_model import NERModel
from pathlib import Path
from typing import Optional
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class NERTestor:
    def __init__(
        self,
        ner_model: NERModel,
        ner_dataset: DatasetDict,
    ) -> None:
        pass
        self.model = ner_model
        self.datasets = ner_dataset
        self.datasets_hash = {
            key: split.__hash__() for key, split in ner_dataset.items()
        }
        self.args = {
            "ner_dataset": self.datasets_hash,
            "ner_model": ner_model.conf,
        }
        self.output_dir = Path("data/output").joinpath(
            md5(str(self.args).encode()).hexdigest()
        )
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

        self.prediction_for_test = self.load_prediction_for("test")
        self.prediction_for_dev = self.load_prediction_for("validation")

        trainer_logger.setLevel(orig_level)

        self.evaluate(self.prediction_for_test)
        self.evaluate_on_head(self.prediction_for_test)
        self.lenient_evaluate(self.prediction_for_test)
        self.evaluate_by_sentence_length(self.prediction_for_test)
        self.evaluate_by_predicted_span_num(self.prediction_for_test)
        self.evaluate_span_detection()

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
        _pred_ner_tags = self.model.batch_predict(tokens)
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
        pred_ner_tags = [
            ["O" if "-fake_cat" in tag else tag for tag in tags]
            for tags in pred_ner_tags
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

        return prediction_for_split

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
        pass

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
        print(ptl.get_string())

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
