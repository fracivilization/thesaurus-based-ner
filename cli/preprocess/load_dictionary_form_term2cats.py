import click
from src.dataset.term2cat.dictionary_form_term2cats import (
    load_dictionary_form_term2cats_jsonl,
)
import tempfile
from src.utils.utils import WeightedSQliteDict


@click.command()
@click.option("--knowledge-base", type=str)
@click.option("--remain-common-sense", type=bool, default=True)
@click.option("--output-dir", type=str)
def main(knowledge_base: str, remain_common_sense: str, output_dir: str):
    work_dir = tempfile.TemporaryDirectory()
    term2cats_jsonl = load_dictionary_form_term2cats_jsonl(
        knowledge_base, remain_common_sense, work_dir=work_dir
    )
    print("term2cats_jsonl is loaded")
    term2cats = WeightedSQliteDict.load_from_jsonl_key_value_weight_triples_file(
        term2cats_jsonl
    )
    term2cats.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
