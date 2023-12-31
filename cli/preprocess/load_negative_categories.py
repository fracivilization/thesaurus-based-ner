from src.dataset.utils import load_negative_cats_from_positive_cats, CATEGORY_SEPARATOR
import click


@click.command()
@click.option("--positive-categories", type=str)
@click.option("--with-negative_categories", type=bool)
@click.option("--eval-dataset", type=str)
def main(positive_categories: str, with_negative_categories: bool, eval_dataset: str):
    # NOTE: スペースでの連結だと複数の単語から構成されるカテゴリ名を利用する際に問題になるかも
    if with_negative_categories:
        positive_categories = positive_categories.split(CATEGORY_SEPARATOR)
        negative_cats = load_negative_cats_from_positive_cats(
            positive_categories, eval_dataset
        )
        print(" ".join(negative_cats))
    else:
        print("")


if __name__ == "__main__":
    main()
