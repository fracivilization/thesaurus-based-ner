from src.dataset.utils import get_umls_negative_cats_from_focus_cats, CATEGORY_SEPARATOR
import click


@click.command()
@click.option("--focus-categories", type=str)
@click.option("--with-negative_categories", type=bool)
def main(focus_categories: str, with_negative_categories: bool):
    if with_negative_categories:
        focus_categories = focus_categories.split(CATEGORY_SEPARATOR)
        print(" ".join(get_umls_negative_cats_from_focus_cats(focus_categories)))
    else:
        print("")


if __name__ == "__main__":
    main()
