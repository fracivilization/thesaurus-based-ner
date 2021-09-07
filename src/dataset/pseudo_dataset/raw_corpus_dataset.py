import os
from pathlib import Path
import shutil
from typing import Dict, Generator, List
import colt
import subprocess
from glob import glob
from datasets import Dataset
from loguru import logger
import spacy
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from p_tqdm import p_umap

pubmed_dir = Path("data/pubmed")
from src.utils.params import pseudo_annotated_time, unlabelled_corpus_dataset_features


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


def load_pubmed_txt_files() -> List[Path]:
    """Return pubmed txt file path"""
    if not pubmed_dir.exists():
        os.mkdir(pubmed_dir)
        for i in range(1, 1063):
            subprocess.run(
                [
                    "wget",
                    "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n{:04d}.xml.gz".format(
                        i
                    ),
                    "-P",
                    pubmed_dir,
                ]
            )
        for gzip_file in glob(str(pubmed_dir.joinpath("pubmed21n*.xml.gz"))):
            subprocess.run(["gunzip", gzip_file])
            subprocess.run(["mv", Path(gzip_file).stem, pubmed_dir])
    files = glob(str(pubmed_dir.joinpath("*.xml")))
    output_files = glob(str(pubmed_dir.joinpath("*.txt")))
    if len(files) != len(output_files):
        # 並列にtextファイル化する
        p_umap(make_pubmed_xml_to_text, files)
        output_files = glob(str(pubmed_dir.joinpath("*.txt")))
    return output_files


@colt.register("RawCorpusDataset")
class RawCorpusDataset:
    def __init__(
        self,
        source: str,
        text_num: int = 0,
        model_name: str = "bert-base-uncased",
        max_length=510,
    ) -> None:
        self.args = {
            "source": source,
            "text_num": text_num,
            "model_name": model_name,
            "max_length": max_length,
        }
        self.source = source
        self.text_num = text_num
        self.model_name = model_name
        self.nlp = None
        self.tokenizer = None
        self.max_length = max_length
        # load txt file for each dataset

    def __post_init__(self):
        # Please specify args
        assert self.args != dict()

    @staticmethod
    def load_txt_files(source) -> Path:
        if source == "pubmed":
            txt_file = load_pubmed_txt_files()
        else:
            raise NotImplementedError
        return txt_file

    @staticmethod
    def load_texts(source: str = "pubmed", text_num: int = -1) -> Generator:
        # todo: テキストファイルにするまでの前処理をRawCorpusでしてしまう
        # 複数のtxtファイルをまとめて、指定された数まで出力するに留める
        files = RawCorpusDataset.load_txt_files(source)
        text_num_tqdm = tqdm(total=text_num)
        for file in files:
            if text_num_tqdm.n == text_num:
                break
            with open(file) as f:
                for text in f:
                    if text_num_tqdm.n == text_num:
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
        if not self.tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )
        tokenizer = self.tokenizer
        nlp = self.nlp
        wide_token_seq = []
        tokenized_wide_token_seq = []
        boss = []
        bos = []
        poss = []
        pos = []
        if len(text) < nlp.max_length:
            doc = nlp(text)
            for snt in doc.sents:
                snt_tokens = [w.text for w in snt]
                tokenized_snt = tokenizer.tokenize(snt_tokens, is_split_into_words=True)
                token_length = len(tokenized_wide_token_seq) + len(tokenized_snt)
                if token_length < self.max_length:
                    bos.append(len(wide_token_seq))
                    pos += [w.tag_ for w in snt]
                    wide_token_seq += snt_tokens
                    tokenized_wide_token_seq += tokenized_snt
                else:
                    tokens.append(wide_token_seq)
                    boss.append(bos)
                    poss.append(pos)
                    wide_token_seq = snt_tokens
                    tokenized_wide_token_seq = tokenized_snt
                    bos = []
                    pos = []
            tokens.append(wide_token_seq)
            boss.append(bos)
            poss.append(pos)
        # else:
        #     if not self.nlp:
        #         self.nlp = spacy.load("en_core_sci_sm")
        #     nlp = self.nlp
        #     if len(text) < nlp.max_length:
        #         doc = nlp(text)
        #         for snt in doc.sents:
        #             tokens.append([w.text for w in snt])
        return {"tokens": tokens, "bos_ids": boss, "POS": poss}

    def load_tokens(self) -> Dataset:
        source = self.source
        text_num = self.text_num
        token_dir = Path(
            os.path.join(
                "data",
                "buffer",
                "tokens_%s_%d" % (source, text_num),
            )
        )

        if token_dir.exists() and token_dir.stat().st_ctime < pseudo_annotated_time:
            # dict_balanced corpusで見つけられた単語に対して事例を収集するように改変　そのためそれ以降に作成されたデータでないときには再度データを作り直す
            shutil.rmtree(token_dir)
            logger.info("file is old, so it is deleted")
        if not token_dir.exists():
            read_snt_num = tqdm(total=text_num)
            tokens = []
            boss = []
            poss = []
            for text in RawCorpusDataset.load_texts(source, text_num):
                tokenized_text = self.tokenize(text)
                assert len(tokenized_text["tokens"]) == len(tokenized_text["bos_ids"])
                assert len(tokenized_text["bos_ids"]) == len(tokenized_text["POS"])
                for tok, bos, pos in zip(
                    tokenized_text["tokens"],
                    tokenized_text["bos_ids"],
                    tokenized_text["POS"],
                ):
                    if read_snt_num.n == text_num:
                        break
                    else:
                        tokens.append(tok)
                        boss.append(bos)
                        poss.append(pos)
                        read_snt_num.update()
                if read_snt_num.n == text_num:
                    break

            pass
            # if there is no pubmed dir
            # nohup wget ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed20n*.xml.gz &
            # gunzip *.gz
            doc_ids = list(range(len(tokens)))
            snt_ids = [-1] * len(tokens)
            tokens = Dataset.from_dict(
                {
                    "tokens": tokens,
                    "doc_id": doc_ids,
                    "snt_id": snt_ids,
                    "bos_ids": boss,
                    "POS": poss,
                },
                features=unlabelled_corpus_dataset_features,
            )
            tokens.save_to_disk(token_dir)
        tokens = Dataset.load_from_disk(token_dir)
        return tokens
