from src.dataset.utils import get_umls_negative_cats_from_focus_cats, CATEGORY_SEPARATOR
import click


@click.command()
@click.option("--focus-categories", type=str)
@click.option("--with-negative_categories", type=bool)
def main(focus_categories: str, with_negative_categories: bool):
    # NOTE: スペースでの連結だと複数の単語から構成されるカテゴリ名を利用する際に問題になるかも
    if with_negative_categories:
        focus_categories = focus_categories.split(CATEGORY_SEPARATOR)
        print(" ".join(get_umls_negative_cats_from_focus_cats(focus_categories)))
    else:
        print("")


if __name__ == "__main__":
    main()
