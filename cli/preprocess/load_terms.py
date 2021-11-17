import click
from src.dataset.term2cat.terms import (
    load_TUI_terms,
    DBPedia_categories,
    load_DBPedia_terms,
)
import re
import os
from tqdm import tqdm
from inflection import UNCOUNTABLES, PLURALS, SINGULARS

PLURAL_RULES = [(re.compile(rule), replacement) for rule, replacement in PLURALS]
SINGULAR_RULES = [(re.compile(rule), replacement) for rule, replacement in SINGULARS]


def pluralize(word: str) -> str:
    """
    Return the plural form of a word.

    Examples::

        >>> pluralize("posts")
        'posts'
        >>> pluralize("octopus")
        'octopi'
        >>> pluralize("sheep")
        'sheep'
        >>> pluralize("CamelOctopus")
        'CamelOctopi'

    """
    if not word or word.lower() in UNCOUNTABLES:
        return word
    else:
        for rule, replacement in PLURAL_RULES:
            if rule.search(word):
                return rule.sub(replacement, word)
        return word


def singularize(word: str) -> str:
    """
    Return the singular form of a word, the reverse of :func:`pluralize`.

    Examples::

        >>> singularize("posts")
        'post'
        >>> singularize("octopi")
        'octopus'
        >>> singularize("sheep")
        'sheep'
        >>> singularize("word")
        'word'
        >>> singularize("CamelOctopi")
        'CamelOctopus'

    """
    for inflection in UNCOUNTABLES:
        if re.search(r"(?i)\b(%s)\Z" % inflection, word):
            return word

    for rule, replacement in SINGULAR_RULES:
        if re.search(rule, word):
            return re.sub(rule, replacement, word)
    return word


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
            for term in tqdm(terms):
                singular = singularize(term)
                plural = pluralize(term)
                f.write("%s\n" % singular)
                f.write("%s\n" % plural)
    else:
        pass


def main():
    cmd()


if __name__ == "__main__":
    main()
