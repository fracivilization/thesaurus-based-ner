from hashlib import md5
import colt
from .raw_corpus_dataset import RawCorpusDataset
from datasets import Dataset, DatasetDict
from pathlib import Path
from seqeval.metrics.sequence_labeling import get_entities
import os
import datasets
from src.utils.params import get_ner_dataset_features, task_name2ner_label_names
from src.ner_model.abstract_model import NERModel
from hashlib import md5
from src.utils.params import pseudo_annotated_time
import shutil
from loguru import logger


@colt.register("PseudoDataset")
class PseudoDataset:
    def __init__(
        self, raw_corpus: RawCorpusDataset, ner_model: NERModel, task: str
    ) -> None:
        self.ner_label_names = ner_model.label_names
        self.unlabelled_corpus = raw_corpus.load_tokens()
        self.ner_model = ner_model
        pass
        self.args = {
            "raw_corpus": raw_corpus.args,
            "ner_model": ner_model.args,
            "task": task,
        }
        self.output_dir = Path("data/buffer").joinpath(
            md5(str(self.args).encode()).hexdigest()
        )
        if (
            self.output_dir.exists()
            and self.output_dir.stat().st_ctime < pseudo_annotated_time
        ):
            # dict_balanced corpusで見つけられた単語に対して事例を収集するように改変　そのためそれ以降に作成されたデータでないときには再度データを作り直す
            shutil.rmtree(self.output_dir)
            logger.info("file is old, so it is deleted")
        logger.info("pseudo annotation output dir: %s" % str(self.output_dir))
        self.datasets = self.load_splitted_dict_matched()

    def load_dictionary_matched_text(self) -> Dataset:
        dict_matched_dir = Path(os.path.join(self.output_dir, "dict_matched"))
        if not dict_matched_dir.exists():
            pre_ner_tagss = self.ner_model.batch_predict(
                self.unlabelled_corpus["tokens"], self.unlabelled_corpus["POS"]
            )
            tokenss, ner_tagss, boss, poss = [], [], [], []
            for tok, bos, pos, nt in zip(
                self.unlabelled_corpus["tokens"],
                self.unlabelled_corpus["bos_ids"],
                self.unlabelled_corpus["POS"],
                pre_ner_tagss,
            ):
                if get_entities(nt):
                    tokenss.append(tok)
                    ner_tagss.append([self.ner_label_names.index(tag) for tag in nt])
                    boss.append(bos)
                    poss.append(pos)
            info = datasets.DatasetInfo(
                features=get_ner_dataset_features(self.ner_label_names)
            )
            doc_id = list(range(len(tokenss)))
            snt_id = [-1] * len(tokenss)
            dict_matched = Dataset.from_dict(
                {
                    "tokens": tokenss,
                    "ner_tags": ner_tagss,
                    "doc_id": doc_id,
                    "snt_id": snt_id,
                    "bos_ids": boss,
                    "POS": poss,
                },
                info=info,
            )
            dict_matched.save_to_disk(dict_matched_dir)
        dict_matched = Dataset.load_from_disk(dict_matched_dir)
        return dict_matched

    def load_splitted_dict_matched(self) -> DatasetDict:
        splitted_dict_matched_dir = Path(
            os.path.join(self.output_dir, "splitted_dict_matched")
        )
        if not splitted_dict_matched_dir.exists():
            self.labeled_snts = self.load_dictionary_matched_text()
            train_validation_split = int(0.9 * len(self.labeled_snts))
            train, validation = (
                self.labeled_snts[:train_validation_split],
                self.labeled_snts[train_validation_split:],
            )
            train = Dataset.from_dict(train, info=self.labeled_snts.info)
            validation = Dataset.from_dict(validation, info=self.labeled_snts.info)
            splitted_dict_matched = DatasetDict(
                {"train": train, "validation": validation}
            )
            splitted_dict_matched.save_to_disk(splitted_dict_matched_dir)
        splitted_dict_matched = DatasetDict.load_from_disk(splitted_dict_matched_dir)
        return splitted_dict_matched
