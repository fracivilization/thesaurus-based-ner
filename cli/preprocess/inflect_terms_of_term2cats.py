import click
from src.utils.utils import WeightedSQliteDict
from tqdm import tqdm
from src.dataset.utils import singularize, pluralize


@click.command()
@click.option("--dictionary-form-term2cats-dir", type=str)
@click.option("--output-path", type=str)
def main(dictionary_form_term2cats_dir, output_path):
    print("term2cats_jsonl is loaded")
    dictionary_form_term2cats = WeightedSQliteDict.load_from_disk(
        dictionary_form_term2cats_dir
    )

    print("load inflected term2cats")
    total_count = len(dictionary_form_term2cats)
    inflected_term2cats = WeightedSQliteDict(output_path, commit_when_set_item=False)
    for term, weighted_cats in tqdm(
        dictionary_form_term2cats.items(), total=total_count
    ):
        inflected_term2cats[term] = weighted_cats

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
                inflected_term2cats[expanded_term] = weighted_cats
    inflected_term2cats.commit()
    del dictionary_form_term2cats
    inflected_term2cats.save_to_disk(output_path)

    # TODO: ログを取れるようにする
    # log_term2cats(inflected_term2cats)


if __name__ == "__main__":
    main()
