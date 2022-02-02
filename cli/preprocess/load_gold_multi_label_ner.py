from chunk import Chunk
import click
from src.dataset.gold_dataset import load_gold_multi_label_ner_datasets
import os
from datasets import DatasetDict
from src.ner_model.chunker import chunker_builder, ChunkerConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig


@click.command()
@click.option("--output-dir", type=str)
@click.option("--input-dir", type=str, default="data/gold/MedMentions/full/data")
def main(output_dir, input_dir):
    if not os.path.exists(output_dir):
        gold_multi_label_ner_datasets: DatasetDict = load_gold_multi_label_ner_datasets(
            input_dir
        )
        gold_multi_label_ner_datasets.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
