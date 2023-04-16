from src.dataset.utils import (
    get_umls_negative_cats_from_positive_cats,
    get_dbpedia_negative_cats_from_positive_cats,
    CATEGORY_SEPARATOR,
)
import click


@click.command()
@click.option("--focus-categories", type=str)
@click.option("--with-negative_categories", type=bool)
@click.option("--eval-dataset", type=str)
def main(focus_categories: str, with_negative_categories: bool, eval_dataset: str):
    # NOTE: スペースでの連結だと複数の単語から構成されるカテゴリ名を利用する際に問題になるかも
    if with_negative_categories:
        focus_categories = focus_categories.split(CATEGORY_SEPARATOR)
        if eval_dataset == "MedMentions":
            negative_cats = get_umls_negative_cats_from_positive_cats(
                focus_categories, eval_dataset
            )
        elif eval_dataset == "CoNLL2003":
            negative_cats = get_dbpedia_negative_cats_from_positive_cats(focus_categories)
        print(" ".join(negative_cats))
    else:
        print("")


if __name__ == "__main__":
    main()
