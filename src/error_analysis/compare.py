from datasets import Dataset
from typing import Union
import os
from typing import List
from ipymarkup.show import show_html
from ipymarkup.palette import palette, BLUE, RED, GREEN, ORANGE, PURPLE
from ipymarkup.span import format_span_ascii_markup
from seqeval.metrics.sequence_labeling import get_entities
from random import choices, seed
from time import time
from pathlib import Path
from loguru import logger


def get_span_line_markup_html(text, spans, **kwargs):
    lines = format_span_ascii_markup(text, spans, **kwargs)
    return lines


def visualize_ner_result(
    tokens: List, base_ner_tags: List, focused_ner_tags: List, gold_ner_tags: List
):
    read_char_num = 0
    tokid2startchar = dict()
    tokid2endchar = dict()
    for tokid, token in enumerate(tokens):
        tokid2startchar[tokid] = read_char_num
        read_char_num += len(token)
        tokid2endchar[tokid] = read_char_num
        read_char_num += 1
    from seqeval.metrics.sequence_labeling import get_entities

    base_spans = [
        (tokid2startchar[s], tokid2endchar[e], "b_" + l)
        for l, s, e in get_entities(base_ner_tags)
    ]
    focused_spans = [
        (tokid2startchar[s], tokid2endchar[e], "f_" + l)
        for l, s, e in get_entities(focused_ner_tags)
    ]
    gold_spans = [
        (tokid2startchar[s], tokid2endchar[e], "g_" + l)
        for l, s, e in get_entities(gold_ner_tags)
    ]
    colors = [BLUE, RED, GREEN, ORANGE, PURPLE]
    palette_args = dict()
    # for l, c in zip(labels, colors):
    #     palette_args["p_%s" % l] = c
    #     palette_args["g_%s" % l] = c
    return "\n".join(
        get_span_line_markup_html(
            " ".join(tokens), base_spans + focused_spans + gold_spans
        )
    )


class CompareNEROutput:
    def __init__(self, base_output: Dataset, focused_output: Dataset):
        labels = set(l for l, s, e in get_entities(base_output["gold_ner_tags"]))
        labels |= set(l for l, s, e in get_entities(base_output["pred_ner_tags"]))
        labels |= set(l for l, s, e in get_entities(focused_output["pred_ner_tags"]))
        labels = list(labels)
        pass
        seed(time())
        output_dir = Path(
            os.path.join(
                "data",
                "output",
                "compare_"
                + "".join(
                    choices(
                        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        k=20,
                    )
                ),
            )
        )
        os.mkdir(output_dir)
        logger.add(output_dir.joinpath("log"))
        logger.info("Compare result is output into %s" % str(output_dir))
        well_over_predicts = []
        bad_over_predicts = []
        bad_over_predicts_span_mismatch = []
        bad_over_predicts_label_mismatch = []
        bad_over_predicts_no_ent = []
        over_predicteds = []
        over_predicted_labels = []
        w_ents_all = []
        g_tags = []
        cf_mx = []  # confusion_matrix
        break_flag = False
        for base_snt, focused_snt in zip(base_output, focused_output):
            #     tokens, _, wo_pred, gold = zip(*wo_snt)
            #     _, _, w_pred, _ = zip(*w_snt)
            tokens = base_snt["tokens"]
            base_pred = base_snt["pred_ner_tags"]
            gold = base_snt["gold_ner_tags"]
            focused_pred = focused_snt["pred_ner_tags"]
            g_tags.append(gold)

            wo_ents = set(get_entities(list(base_pred)))
            w_ents = set(get_entities(list(focused_pred)))
            g_ents = set(get_entities(list(gold)))
            w_ents_all += list(w_ents)
            assert len(base_pred) == len(gold) and len(gold) == len(focused_pred)
            more_predicteds = w_ents - wo_ents
            for l, s, e in more_predicteds:
                stt = max(0, s - 5)
                ed = e + 6
                over_predicteds.append(
                    visualize_ner_result(
                        list(tokens)[stt:ed],
                        list(base_pred)[stt:ed],
                        list(focused_pred)[stt:ed],
                        list(gold)[stt:ed],
                    )
                )
                over_predicted_labels.append(l)
            gold_words = set([i for l, s, e in g_ents for i in range(s, e + 1)])
            for l, s, e in more_predicteds & g_ents:
                stt = max(0, s - 5)
                ed = e + 6
                well_over_predicts.append(
                    visualize_ner_result(
                        list(tokens)[stt:ed],
                        list(base_pred)[stt:ed],
                        list(focused_pred)[stt:ed],
                        list(gold)[stt:ed],
                    )
                )
            for l, s, e in more_predicteds - g_ents:
                start = max(0, s - 5)
                end = e + 6
                #         print(l,s,e)
                span_words = set(range(s, e + 1))
                if span_words & gold_words:
                    gl, gs, ge = [
                        (gl, gs, ge)
                        for gl, gs, ge in g_ents
                        if set(range(gs, ge + 1)) & set(range(s, e + 1))
                    ][0]
                    cf_mx.append((l, gl))
                    if not l.startswith("fake_cat_"):
                        if gl == l:
                            bad_over_predicts_span_mismatch.append(
                                visualize_ner_result(
                                    list(tokens)[start:end],
                                    list(base_pred)[start:end],
                                    list(focused_pred)[start:end],
                                    list(gold)[start:end],
                                )
                            )
                        # if ラベルが一致していれば
                        # 重複があるentityの図示を bad_over_predicts_span_mismatch に 追加する
                        else:
                            bad_over_predicts_label_mismatch.append(
                                visualize_ner_result(
                                    list(tokens)[start:end],
                                    list(base_pred)[start:end],
                                    list(focused_pred)[start:end],
                                    list(gold)[start:end],
                                )
                            )
                        # 重複があるentityの図示を bad_over_predicts_label_mismatch に 追加する
                        pass
                else:
                    if not l.startswith("fake_cat_"):
                        bad_over_predicts_no_ent.append(
                            visualize_ner_result(
                                list(tokens)[start:end],
                                list(base_pred)[start:end],
                                list(focused_pred)[start:end],
                                list(gold)[start:end],
                            )
                        )
                        # bad_over_predicts_no_ent に entityの図示を追加する
                pass
            if break_flag:
                break
        x = 2
        logger.info("# well over predicts: %d" % len(well_over_predicts))
        with open(output_dir.joinpath("well_over_predicts.txt"), "w") as f:
            f.write("\n".join(well_over_predicts))
        logger.info(
            "# bad over predicts: %d"
            % len(
                bad_over_predicts_span_mismatch
                + bad_over_predicts_label_mismatch
                + bad_over_predicts_no_ent
            )
        )
        logger.info(
            "# bad over predicts by span mismatch: %d"
            % len(bad_over_predicts_span_mismatch)
        )
        with open(output_dir.joinpath("bad_over_predicts_span_mismatch.txt"), "w") as f:
            f.write("\n".join(bad_over_predicts_span_mismatch))
        logger.info(
            "# bad over predicts by label mismatch: %d"
            % len(bad_over_predicts_label_mismatch)
        )
        with open(
            output_dir.joinpath("bad_over_predicts_label_mismatch.txt"), "w"
        ) as f:
            f.write("\n".join(bad_over_predicts_label_mismatch))
        logger.info(
            "# bad over predicts by no entity: %d" % len(bad_over_predicts_no_ent)
        )
        with open(output_dir.joinpath("bad_over_predicts_no_ent.txt"), "w") as f:
            f.write("\n".join(bad_over_predicts_no_ent))
