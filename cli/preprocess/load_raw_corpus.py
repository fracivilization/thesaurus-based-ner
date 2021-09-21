#!/usr/bin/env python
# -*- coding: utf-8 -*-
import click
from src.dataset.raw_corpus import RawCorpusDataset
import os
from logging import getLogger

logger = getLogger(__name__)


@click.command()
@click.option("--source-txt-dir", default="data/raw/pubmed")
@click.option("--raw-sentence-num", type=int, default=10000)
@click.option("--output-dir", default=".")
def cmd(raw_sentence_num: int, output_dir: str, source_txt_dir: str):
    """Load txt files in source_txt_dir and Split by sentence and words and save into output dir
    load sentences for raw_sentence_num

    Args:
        raw_corpus_num (int): [description]
        output_dir (str): [description]
        source_txt_dir (str): [description]
    """
    if not os.path.exists(output_dir):
        raw_corpus = RawCorpusDataset(source_txt_dir, raw_sentence_num)
        tokens = raw_corpus.load_tokens()
        tokens.save_to_disk(output_dir)
    tokens.load_from_disk(output_dir)
    logger.info(tokens.info.description)


def main():
    cmd()


if __name__ == "__main__":
    main()
