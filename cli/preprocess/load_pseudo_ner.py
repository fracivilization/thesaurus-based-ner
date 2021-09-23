import click
from src.dataset.term2cat.terms import (
    load_TUI_terms,
    DBPedia_categories,
    load_DBPedia_terms,
)
import re
import os


@click.command()
@click.option(
    "--raw-corpus",
    type=str,
    default="data/raw/adc83b19e793491b1c6ea0fd8b46cd9f32e592fc",
)
@click.option("--focus-cats", type=str, default="T116_T126")
@click.option(
    "--duplicate-cats",
    type=str,
    default="ChemicalSubstance_GeneLocation_Biomolecule_Unknown_TopicalConcept",
)
@click.option(
    "--output-dir",
    type=str,
    default="data/pseudo/dbc9968f1dd87e3bfab1fee210700a95ebfa65e1",
)
@click.option(
    "--gold-corpus",
    type=str,
    default="data/gold/7bd600d361001d5acc3b1e3f2974b2536027ea20",
)
def cmd(
    raw_corpus: str,
    focus_cats: str,
    duplicate_cats: str,
    output_dir: str,
    gold_corpus: str,
):
    pass


def main():
    cmd()


if __name__ == "__main__":
    main()
