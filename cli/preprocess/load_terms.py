import click
from src.dataset.term2cat.terms import (
    load_TUI_terms,
    load_DBPedia_terms,
    CoNLL2003Categories,
    load_CoNLL2003_terms,
)
import re
import os
from tqdm import tqdm
from src.dataset.utils import singularize, pluralize, load_DBPediaCategories


@click.command()
@click.option("--category", type=str, default="T116")
@click.option("--output", type=str, default="data/dict/T116")
def cmd(category: str, output: str):
    pattern = "T\d{3}"
    DBPediaCategories = load_DBPediaCategories()
    if re.match(pattern, category):
        terms = load_TUI_terms(category)
    elif category in DBPediaCategories:
        terms = load_DBPedia_terms([category])
    elif category in CoNLL2003Categories:
        terms = load_CoNLL2003_terms(category)
    else:
        raise NotImplementedError
    with open(output, "w") as f:
        for term in tqdm(terms):
            singular = singularize(term)
            plural = pluralize(term)
            f.write("%s\n" % singular)
            f.write("%s\n" % plural)


def main():
    cmd()


if __name__ == "__main__":
    main()
