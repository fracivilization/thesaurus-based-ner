import os
from pathlib import Path
import shutil
from typing import Dict, Generator, List
from glob import glob
from datasets import Dataset
import spacy
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from p_tqdm import p_umap
from src.utils.params import pseudo_annotated_time, unlabelled_corpus_dataset_features
from logging import getLogger
import datasets
import json

logger = getLogger(__name__)


def make_pubmed_xml_to_text(file):
    output_file = Path(file[: file.rfind(".xml")] + ".txt")
    if not output_file.exists():
        import xml.etree.ElementTree as ET

        tree = ET.parse(file)
        root = tree.getroot()
        texts = []
        for abstract_text in root.findall(
            "./PubmedArticle/MedlineCitation/Article/Abstract/AbstractText"
        ):
            if abstract_text.text:
                texts += [abstract_text.text]
        with open(output_file, "w") as f:
            f.write("\n".join(texts))


class RawCorpusDataset:
    def __init__(
        self,
        source_txt_dir: str,
        sentence_num: int = 0,
    ) -> None:
        self.args = {
            "source_txt_dir": source_txt_dir,
            "sentence_num": sentence_num,
        }
        self.source_txt_dir = source_txt_dir
        self.sentence_num = sentence_num
        self.nlp = None
        # load txt file for each dataset

    def load_txt_files(self) -> Path:
        return glob(os.path.join(self.source_txt_dir, "*.txt"))

    def load_texts(self) -> Generator:
        # todo: テキストファイルにするまでの前処理をRawCorpusでしてしまう
        # 複数のtxtファイルをまとめて、指定された数まで出力するに留める
        files = self.load_txt_files()
        text_num_tqdm = tqdm(total=self.sentence_num)
        for file in files:
            if text_num_tqdm.n == self.sentence_num:
                break
            with open(file) as f:
                for text in f:
                    if text_num_tqdm.n == self.sentence_num:
                        break
                    else:
                        yield text
                        text_num_tqdm.update()

    def tokenize(self, text: str) -> Dict:
        tokens = []
        # if self.args.use_wide_context:
        # todo: bertのtokenizer読み込む
        if not self.nlp:
            self.nlp = spacy.load("en_core_sci_sm")
            # todo: bert tokenizerってsentence splitしてくれるんだっけ？
        nlp = self.nlp
        if len(text) < nlp.max_length:
            doc = nlp(text)
            for snt in doc.sents:
                snt_tokens = [w.text for w in snt]
                tokens.append(snt_tokens)
        return {"tokens": tokens}

    def load_tokens(self) -> Dataset:
        read_snt_num = tqdm(total=self.sentence_num)
        tokens = []
        for text in self.load_texts():
            text = text.strip()
            tokenized_text = self.tokenize(text)
            assert len(tokenized_text["tokens"])
            for tok in tokenized_text["tokens"]:
                if read_snt_num.n == self.sentence_num:
                    break
                else:
                    tokens.append(tok)
                    read_snt_num.update()
            if read_snt_num.n == self.sentence_num:
                break
        tokens = Dataset.from_dict(
            {"tokens": tokens},
            info=datasets.info.DatasetInfo(
                features=datasets.Features(
                    {"tokens": datasets.Sequence(datasets.Value("string"))}
                ),
                description=json.dumps(self.args),
            ),
        )
        return tokens
