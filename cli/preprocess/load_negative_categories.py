from src.dataset.utils import get_umls_negative_cats_from_focus_cats, CATEGORY_SEPARATOR
import click


@click.command()
@click.option("--focus-categories", type=str)
def main(focus_categories: str):
    focus_categories = focus_categories.split(CATEGORY_SEPARATOR)
    print("_".join(get_umls_negative_cats_from_focus_cats(focus_categories)))


if __name__ == "__main__":
    main()
