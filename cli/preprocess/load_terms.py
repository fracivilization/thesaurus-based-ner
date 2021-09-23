import click
from src.dataset.term2cat.terms import (
    load_TUI_terms,
    DBPedia_categories,
    load_DBPedia_terms,
)
import re
import os


@click.command()
@click.option("--category", type=str, default="T116")
@click.option("--output", type=str, default="data/dict/T116")
def cmd(category: str, output: str):
    pattern = "T\d{3}"
    if not os.path.exists(output):
        if re.match(pattern, category):
            terms = load_TUI_terms(category)
        elif category in DBPedia_categories:
            terms = load_DBPedia_terms(category)
        else:
            raise NotImplementedError
        with open(output, "w") as f:
            f.write("\n".join(terms))
    else:
        pass


def main():
    cmd()


if __name__ == "__main__":
    main()
