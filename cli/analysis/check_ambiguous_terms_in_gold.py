from collections import defaultdict
from typing import Counter
import click
from datasets import DatasetDict, Dataset
from seqeval.metrics.sequence_labeling import get_entities
from prettytable import PrettyTable


def show_umbiguous_terms_statistics(split: Dataset, lower=False):
    ner_tag_names = split.features["ner_tags"].feature.names
    mention2labels = defaultdict(list)
    mention2count = Counter()
    terms = set()
    mentions = list()
    for snt in split:
        tokens = snt["tokens"]
        ner_tags = [ner_tag_names[tag] for tag in snt["ner_tags"]]
        for l, s, e in get_entities(ner_tags):
            term = " ".join(tokens[s : e + 1])
            if lower:
                term = term.lower()
            mentions.append(term)
            mention2count[term] += 1
            terms.add(term)
            mention2labels[term].append(l)
    umbiguous_terms = [
        (term, set(labels))
        for term, labels in mention2labels.items()
        if len(set(labels)) > 1
    ]
    tab = PrettyTable()
    tab.field_names = ["", "Umbiguous", "All", "Ratio (%)"]
    umbiguous_mention_count = sum(
        [len(mention2labels[term]) for term, labels in umbiguous_terms]
    )
    tab.add_rows(
        [
            [
                "Term",
                len(umbiguous_terms),
                len(terms),
                "%.2f" % (len(umbiguous_terms) / len(terms) * 100,),
            ],
            [
                "Mention",
                umbiguous_mention_count,
                len(mentions),
                "%.2f" % (umbiguous_mention_count / len(mentions) * 100,),
            ],
        ]
    )
    print(tab)


@click.command()
@click.option(
    "--gold_data_path",
    help="Path of gold dataset taken by make make_gold_ner_data",
    default="data/gold/836053d1bd47c7cf824672a17714a61a354af8e9",
)
def main(gold_data_path):
    gold_data = DatasetDict.load_from_disk(gold_data_path)
    for key, split in gold_data.items():
        print(key)
        show_umbiguous_terms_statistics(split)
        print("lowercased")
        show_umbiguous_terms_statistics(split, lower=True)
    print(gold_data_path)


if __name__ == "__main__":
    main()
