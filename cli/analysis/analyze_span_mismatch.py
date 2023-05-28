import click
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict
from typing import List
from prettytable import PrettyTable


def get_tokid2included_span(ner_tags: List[str]):
    tokid2span = dict()
    for l, s, e in get_entities(ner_tags):
        for i in range(s, e + 1):
            tokid2span[i] = (l, s, e)
    return tokid2span


@click.command()
@click.option("--input-file", "-i", help="Input CoNLL formatted file.")
def cli(input_file):
    """Analyze span mismatch."""

    with open(input_file, "r") as f:
        snts = f.read().split("\n\n")
    snts = [[word.split("\t") for word in snt.split("\n")] for snt in snts]
    snts[0] = snts[0][1:]

    # 予測されたスパンを
    # - スパン・ラベル完全一致
    # - ラベルが一致する部分重複のスパンがない
    # - スパン不一致に分ける
    # さらにスパン不一致に関しては
    # - スパン末端で不足
    # - スパン末端で超過
    # - スパン開始で超過
    # - スパン開始で不足
    # に分ける、というのを全体と各予測ラベルについて行う
    labels = set()
    pred_label_num = 0
    tp_num = 0
    fp_num = 0
    tp_num_by_label = defaultdict(int)
    fp_num_by_label = defaultdict(int)
    early_end_num = 0
    early_end_num_by_label = defaultdict(int)
    late_end_num = 0
    late_end_num_by_label = defaultdict(int)
    early_start_num = 0
    early_start_num_by_label = defaultdict(int)
    late_start_num = 0
    late_start_num_by_label = defaultdict(int)

    for snt in snts:
        _, pred, gold = map(list, zip(*snt))
        pred_spans = get_entities(pred)
        gold_spans = get_entities(gold)
        tokid2gold_span = get_tokid2included_span(gold)
        pred_label_num += len(pred_spans)
        for pred_span in pred_spans:
            pl, ps, pe = pred_span
            labels.add(pl)
            if pred_span in gold_spans:
                tp_num += 1
                tp_num_by_label[pl] += 1
            else:
                partial_matched_spans = set()
                for tokid in range(ps, pe + 1):
                    if tokid in tokid2gold_span:
                        gold_sapn = tokid2gold_span[tokid]
                        if pl == gold_sapn[0]:
                            partial_matched_spans.add(gold_sapn)

                if partial_matched_spans:
                    for _, gs, ge in partial_matched_spans:
                        if gs > ps:
                            early_start_num += 1
                            early_start_num_by_label[pl] += 1
                        elif gs < ps:
                            late_start_num += 1
                            late_start_num_by_label[pl] += 1

                        if ge > pe:
                            early_end_num += 1
                            early_end_num_by_label[pl] += 1
                        elif ge < pe:
                            late_end_num += 1
                            late_end_num_by_label[pl] += 1
                else:
                    fp_num += 1
                    fp_num_by_label[pl] += 1

            pred_span = range(ps, pe + 1)

    tbl = PrettyTable(
        ["Label", "TP", "FP", "Early Start", "Late Start", "Early End", "Late End"]
    )
    for label in labels:
        tbl.add_row(
            [
                label,
                tp_num_by_label[label],
                fp_num_by_label[label],
                early_start_num_by_label[label],
                late_start_num_by_label[label],
                early_end_num_by_label[label],
                late_end_num_by_label[label],
            ]
        )
    tbl.add_row(
        [
            "All",
            tp_num,
            fp_num,
            early_start_num,
            late_start_num,
            early_end_num,
            late_end_num,
        ]
    )
    print(tbl.get_string(title="Span Mismatch Analysis"))


if __name__ == "__main__":
    cli()
