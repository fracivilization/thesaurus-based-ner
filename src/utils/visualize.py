from typing import List
from ipymarkup.show import show_html
from ipymarkup.palette import palette, BLUE, RED, GREEN, ORANGE, PURPLE
from ipymarkup.span import format_span_line_markup


def get_span_line_markup_html(text, spans, **kwargs):
    lines = format_span_line_markup(text, spans, **kwargs)
    return lines


def visualize_ner_result(
    tokens: List[str], pred_ner_tags: List[str], gold_ner_tags: List[str]
):
    labels = [
        "protein",
        "DNA",
        "RNA",
        "cell_type",
        "cell_line",
    ]  # todo: revise this line if you use non GENIA dataset
    read_char_num = 0
    tokid2startchar = dict()
    tokid2endchar = dict()
    for tokid, token in enumerate(tokens):
        tokid2startchar[tokid] = read_char_num
        read_char_num += len(token)
        tokid2endchar[tokid] = read_char_num
        read_char_num += 1
    from seqeval.metrics.sequence_labeling import get_entities

    pred_spans = [
        (tokid2startchar[s], tokid2endchar[e], "p_" + l)
        for l, s, e in get_entities(pred_ner_tags)
    ]
    gold_spans = [
        (tokid2startchar[s], tokid2endchar[e], "g_" + l)
        for l, s, e in get_entities(gold_ner_tags)
    ]
    colors = [BLUE, RED, GREEN, ORANGE, PURPLE]
    palette_args = dict()
    for l, c in zip(labels, colors):
        palette_args["p_%s" % l] = c
        palette_args["g_%s" % l] = c
    return list(
        get_span_line_markup_html(
            " ".join(tokens), pred_spans + gold_spans, palette=palette(**palette_args)
        )
    )
