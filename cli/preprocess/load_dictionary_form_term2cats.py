import click
from src.dataset.term2cat.dictionary_form_term2cats import (
    load_dictionary_form_term_cats_weights,
)
import tempfile
from src.utils.utils import WeightedSQliteDict, WeightedValues


@click.command()
@click.option("--knowledge-base", type=str)
@click.option("--remain-common-sense", type=bool, default=True)
@click.option("--output-dir", type=str)
def main(knowledge_base: str, remain_common_sense: str, output_dir: str):
    work_dir = tempfile.TemporaryDirectory()
    term2cats = WeightedSQliteDict(output_dir, commit_when_set_item=False)
    for term, cats, weights in load_dictionary_form_term_cats_weights(
        knowledge_base, remain_common_sense, work_dir=work_dir
    ):
        term2cats[term] = WeightedValues(cats, weights)
    term2cats.save_to_disk(output_dir)


if __name__ == "__main__":
    main()
