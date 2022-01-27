from chunk import Chunk
import click
from src.dataset.gold_dataset import load_gold_msmlc_datasets
import os
from datasets import DatasetDict
from src.ner_model.chunker import chunker_builder, ChunkerConfig
from src.ner_model.chunker.spacy_model import SpacyNPChunkerConfig


@click.command()
@click.option("--output-dir", type=str)
@click.option("--input-dir", type=str, default="data/gold/MedMentions/full/data")
@click.option("--with-o", type=bool, default=True)
@click.option("--chunker", type=str, default="enumerated")
def main(output_dir, input_dir, with_o, chunker):
    if not os.path.exists(output_dir):
        if chunker == "spacy_np":
            chunker_config = SpacyNPChunkerConfig()
        else:
            raise NotImplementedError
        chunker_for_o = chunker_builder(chunker_config)
        gold_msmlc_datasets: DatasetDict = load_gold_msmlc_datasets(
            input_dir, with_o, chunker_for_o
        )
        gold_msmlc_datasets.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
