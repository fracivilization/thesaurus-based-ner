import colt
from datasets import DatasetDict
from seqeval.metrics.sequence_labeling import get_entities
from collections import Counter
from loguru import logger
from src.utils.params import get_ner_dataset_features, get_ner_labels
from datasets import Dataset, DatasetDict


@colt.register("PseudoAnnoAddedDataset")
class PseudoAnnoAddedDataset:
    def __new__(self, ner_dataset: DatasetDict, ner_model) -> None:
        # TODO: feature を作成する
        orig_tag_names = ner_dataset["train"].features["ner_tags"].feature.names
        orig_labels = [l for l, s, e in get_entities(orig_tag_names)]
        new_labels = [l for l, s, e in get_entities(ner_model.label_names)]
        features = get_ner_dataset_features(
            get_ner_labels(sorted(orig_labels + new_labels))
        )
        new_dataset_dict = dict()
        for key, split in ner_dataset.items():
            if key in {"train", "validation"}:
                orig_dataset = {k: split[k] for k in split.features}
                pred_tags = ner_model.batch_predict(split["tokens"])
                # ner_model.batch_predict()
                logger.info(Counter(l for l, s, e in get_entities(pred_tags)))
                orig_tags = [
                    [orig_tag_names[tag] for tag in snt] for snt in split["ner_tags"]
                ]
                new_tags = []
                for snt_id, (orig_tag, pred_tag) in enumerate(
                    zip(orig_tags, pred_tags)
                ):
                    orig_ents = get_entities(orig_tag)
                    orig_tagged_words = set(
                        [i for l, s, e in orig_ents for i in range(s, e + 1)]
                    )
                    for l, s, e in get_entities(pred_tag):
                        pred_tagged_words = set(range(s, e + 1))
                        if not orig_tagged_words & pred_tagged_words:
                            for i in range(s, e + 1):
                                if i == s:
                                    orig_tag[i] = "B-%s" % l
                                else:
                                    orig_tag[i] = "I-%s" % l
                    new_tags.append(orig_tag)
                orig_dataset["ner_tags"] = new_tags
                new_dataset_dict[key] = Dataset.from_dict(
                    orig_dataset, features=features
                )
            elif key == "test":
                # ner_tagを変換する
                orig_dataset = {k: split[k] for k in split.features}
                orig_tags = [
                    [orig_tag_names[tag] for tag in snt] for snt in split["ner_tags"]
                ]
                orig_dataset["ner_tags"] = orig_tags
                new_dataset_dict[key] = Dataset.from_dict(
                    orig_dataset, features=features
                )
            else:
                raise NotImplementedError
        return DatasetDict(new_dataset_dict)
