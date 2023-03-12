import click
import tempfile
from src.utils.utils import DoubleArrayDictWithIterators
import os
from tqdm import tqdm
import json
from src.dataset.utils import singularize, pluralize
from src.dataset.term2cat.dictionary_form_term2cats import (
    log_term2cats,
)


@click.command()
@click.option("--dictionary-form-term2cats-dir", type=str)
@click.option("--output-dir", type=str)
def main(dictionary_form_term2cats_dir, output_dir):
    print("term2cats_jsonl is loaded")
    dictionary_form_term2cats = DoubleArrayDictWithIterators.load_from_disk(
        dictionary_form_term2cats_dir
    )
    work_dir = tempfile.TemporaryDirectory()

    print("load inflected term2cats")
    inflected_term2cats_file = os.path.join(work_dir.name, "inflected_term2cats.jsonl")
    total_count = sum(1 for _ in dictionary_form_term2cats.items())
    with open(inflected_term2cats_file, "w") as f_output:
        for term, cats in tqdm(dictionary_form_term2cats.items(), total=total_count):
            f_output.write(json.dumps([term, cats]) + "\n")

            singularized_term = singularize(term)
            pluralized_term = pluralize(term)

            expanded_terms = set()
            for new_term in [singularized_term, pluralized_term]:
                if new_term != term:
                    expanded_terms.add(new_term)

            for expanded_term in expanded_terms:
                if expanded_term == "":
                    # NOTE: 元の単語が's'の場合拡張することによって空文字になる可能性がある
                    break
                if expanded_term not in dictionary_form_term2cats:
                    f_output.write(json.dumps([expanded_term, cats]) + "\n")

    del dictionary_form_term2cats
    inflected_term2cats = DoubleArrayDictWithIterators(inflected_term2cats_file)
    inflected_term2cats.save_to_disk(output_dir)

    # TODO: ログを取れるようにする
    # log_term2cats(inflected_term2cats)


if __name__ == "__main__":
    main()
