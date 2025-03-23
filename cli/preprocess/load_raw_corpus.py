#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click
from src.dataset.raw_corpus import RawCorpusDataset
from logging import getLogger
import sys

logger = getLogger(__name__)


@click.command()
@click.option("--source-txt", type=str, required=True)
@click.option("--raw-sentence-num", type=int, default=sys.maxsize)
@click.option("--output-dir", default=".")
@click.option("--spacy-model", default="en_core_sci_sm")
def cmd(raw_sentence_num: int, output_dir: str, source_txt: str, spacy_model: str):
    """Load txt files in source_txt_dir and Split by sentence and words and save into output dir
    load sentences for raw_sentence_num

    Args:
        raw_corpus_num (int): [description]
        output_dir (str): [description]
        source_txt_dir (str): [description]
    """
    raw_corpus = RawCorpusDataset(source_txt, raw_sentence_num, spacy_model)
    tokens = raw_corpus.load_tokens()
    tokens.save_to_disk(output_dir)

    logger.info(tokens.info.description)


def main():
    cmd()


if __name__ == "__main__":
    main()
