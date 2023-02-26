from .abstract import MultiLabelTyper
from datasets import DatasetDict, Dataset
from src.utils.mlflow import MlflowWriter
from pathlib import Path
from loguru import logger
import os
from scipy.special import expit


class MultiLabelTestor:
    "NER Testor test NER Mondel on dataset"

    def __init__(
        self,
        multi_label_typer_model: MultiLabelTyper,
        msml_dataset: DatasetDict,
        writer: MlflowWriter,
        # config: NERTestorConfig,
    ) -> None:
        pass
        # For debugging
        msml_dataset = DatasetDict(
            {
                key: Dataset.from_dict(split[:1000], features=split.features)
                for key, split in msml_dataset.items()
            }
        )
        self.multi_label_typer_model = multi_label_typer_model
        self.datasets = msml_dataset
        focused_cats = (
            msml_dataset["validation"].features["labels"].feature.feature.names
            + msml_dataset["test"].features["labels"].feature.feature.names
        )
        self.focused_cats = set(focused_cats)
        self.datasets_hash = {
            key: split.__hash__() for key, split in msml_dataset.items()
        }
        self.args = {
            "ner_dataset": self.datasets_hash,
            "ner_model": multi_label_typer_model.conf,
        }
        self.output_dir = Path(".")
        self.writer = writer
        logger.info("output_dir for NERTestor: %s" % str(self.output_dir))
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)
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

    def load_prediction_for(self, split="test"):
        # transformersのlogging level変更

        # 変数初期化
        gold_labels = []
        label_names = self.datasets[split].features["labels"].feature.feature.names
        logger.info("Start predictions for %s." % split)
        tokens = self.datasets[split]["tokens"]
        starts = self.datasets[split]["starts"]
        ends = self.datasets[split]["ends"]
        # _dict_ner_tags = self.dict_match_ner_model.batch_predict(tokens)
        _pred_ner_tags = self.multi_label_typer_model.batch_predict(
            tokens, starts, ends
        )
        # pred_ner_tagsから fake_cat_で始まるラベルをのぞく
        _gold_ner_tags = [
            [[label_names[label] for label in labels] for labels in snt["labels"]]
            for snt in self.datasets[split]
        ]
        tokens = []
        pred_labels = []
        pred_probs = []
        # dict_ner_tags = []
        gold_labels = []
        for doc, _pred, _gold in zip(
            self.datasets[split],
            _pred_ner_tags,
            _gold_ner_tags,
            #  _dict_ner_tags
        ):
            tokens.append(doc["tokens"])
            pred_labels.append([span.labels for span in _pred])
            pred_probs.append([expit(span.logits) for span in _pred])
            gold_labels.append(_gold)
            # dict_ner_tags.append(_dict)
        assert len(tokens) == len(gold_labels)
        assert len(gold_labels) == len(pred_labels)
        # assert len(gold_ner_tags) == len(dict_ner_tags)
        prediction_for_split = Dataset.from_dict(
            {
                "tokens": tokens,
                "starts": starts,
                "ends": ends,
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "pred_probs": pred_probs
                # "dict_ner_tags": dict_ner_tags,
            }
        )
        prediction_for_split.save_to_disk(
            os.path.join(self.output_dir, "prediction_for_%s.dataset" % split)
        )

        return prediction_for_split

    def evaluate(self, prediction_for_test: Dataset):
        logger.info("evaluate strict ner score")
        from sklearn.metrics import (
            precision_recall_fscore_support,
            precision_score,
            recall_score,
            f1_score,
        )

        # logger.info(
        #     classification_report(
        #         prediction_for_test["gold_labels"],
        #         prediction_for_test["pred_labels"],
        #         digits=4,
        #     )
        # )
        logger.info("| cat | P. | R. | F. |")
        for cat in sorted(self.focused_cats):
            gold_labels = [
                int(cat in span_labels)
                for snt_labels in prediction_for_test["gold_labels"]
                for span_labels in snt_labels
            ]
            pred_labels = [
                int(cat in span_labels)
                for snt_labels in prediction_for_test["pred_labels"]
                for span_labels in snt_labels
            ]
            P = precision_score(gold_labels, pred_labels)
            R = recall_score(gold_labels, pred_labels)
            F = f1_score(gold_labels, pred_labels)
            logger.info(
                "| %s | %.2f | %.2f | %.2f |" % (cat, 100 * P, 100 * R, 100 * F)
            )
        # self.writer.log_metric("precision", 100 * P)
        # self.writer.log_metric("recall", 100 * R)
        # self.writer.log_metric("f1", 100 * F)
