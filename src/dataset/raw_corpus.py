import os
from pathlib import Path
from typing import Dict, Generator
from glob import glob
from datasets import Dataset
import spacy
from tqdm import tqdm
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
        source_txt: str,
        sentence_num: int = 0,
        spacy_model="en_core_sci_sm",
    ) -> None:
        self.args = {
            "source_txt": source_txt,
            "sentence_num": sentence_num,
        }
        self.source_txt = source_txt
        self.sentence_num = sentence_num
        self.nlp = spacy.load(spacy_model)
        # load txt file for each dataset

    def load_texts(self) -> Generator:
        text_num_tqdm = tqdm(total=self.sentence_num)
        with Path(self.source_txt).open() as f:
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
            if not tokenized_text["tokens"]:
                continue
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
