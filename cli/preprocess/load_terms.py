import click
from src.dataset.term2cat.terms import (
    load_PubChem_terms,
    load_TUI_terms,
    DBPedia_categories,
    load_DBPedia_terms,
)
import re
import os
from tqdm import tqdm
from src.dataset.utils import singularize, pluralize

@click.command()
@click.option("--category", type=str, default="T116")
@click.option("--output", type=str, default="data/dict/T116")
def cmd(category: str, output: str):
    pattern = "T\d{3}"
    if re.match(pattern, category):
        terms = load_TUI_terms(category)
    elif category in DBPedia_categories:
        terms = load_DBPedia_terms(category)
    elif category == "PubChem":
        terms = load_PubChem_terms()
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
