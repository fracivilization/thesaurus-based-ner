from typing import List, Dict
from scipy.special import softmax
import numpy as np
from src.dataset.utils import CoNLL2003CategoryMapper

def load_category_mapped_focus_and_negative_cats(category_mapper, used_logit_label_names):
    category_mapped_focus_and_negative_cats = []
    for cat in used_logit_label_names:
        if cat in category_mapper:
            category_mapped_focus_and_negative_cats.append(category_mapper[cat])
        else:
            category_mapped_focus_and_negative_cats.append(cat)
    return list(
        sorted(set(category_mapped_focus_and_negative_cats))
    )

def load_ner_tags(starts, ends, labels, probs, snt_len):
    labeled_chunks = sorted(
        zip(starts, ends, labels, probs),
        key=lambda x: x[3],
        reverse=True,
    )
    ner_tags = ["O"] * snt_len
    for s, e, label, max_prob in labeled_chunks:
        if not label.startswith("nc-") and all(tag == "O" for tag in ner_tags[s:e]):
            for i in range(s, e):
                if i == s:
                    ner_tags[i] = "B-%s" % label
                else:
                    ner_tags[i] = "I-%s" % label
    assert snt_len == len(ner_tags)
    return ner_tags

def extract_focus_and_negative_cats(starts, ends, outputs, 
                                    used_logit_label_names, used_logit_label_ids, category_mapper,
                                    category_mapped_focus_and_negative_cats, negative_cats):
    remained_starts = []
    remained_ends = []
    remained_labels = []
    max_probs = []
    for s, e, o in zip(starts, ends, outputs):
        # positive_catsが何かしら出力されているスパンのみ残す
        # 更に残っている場合は最大確率のものを残す
        focus_and_negative_prob = softmax(o.logits[used_logit_label_ids])
        if category_mapper:
            category_mapped_probs = [0] * len(category_mapped_focus_and_negative_cats)
            assert len(used_logit_label_names) == len(focus_and_negative_prob)
            for cat, prob in zip(used_logit_label_names, focus_and_negative_prob):
                if category_mapper:
                    if cat in category_mapper:
                        cat = category_mapper[cat]
                    cat_id = category_mapped_focus_and_negative_cats.index(cat)
                    category_mapped_probs[cat_id] += prob
            label_names = category_mapped_focus_and_negative_cats
            focus_and_negative_prob = np.array(category_mapped_probs)
        else:
            label_names = used_logit_label_names
        max_prob = focus_and_negative_prob.max()
        label = label_names[focus_and_negative_prob.argmax()]
        if label in negative_cats:
            remained_labels.append("nc-%s" % label)
        else:
            remained_labels.append(label)
        remained_starts.append(s)
        remained_ends.append(e)
        max_probs.append(max_prob)
    return remained_starts, remained_ends, remained_labels, max_probs

def load_information_to_extract_focus_and_negative_cats(
        ml_label_names: List[str],
        positive_cats: List[str],
        negative_cats: List[str]):
    focus_and_negative_cats = ["nc-O"] + positive_cats + negative_cats
    category_mapper = dict()
    used_logit_label_names = []
    used_logit_label_ids = []
    for label in focus_and_negative_cats:
        if label in CoNLL2003CategoryMapper:
            for category in CoNLL2003CategoryMapper[label]:
                # NOTE: 学習データ中に出現しなかったラベルは無視する
                category_mapper[category] = label
                used_logit_label_names.append(category)
                used_logit_label_ids.append(ml_label_names.index(category))
        # NOTE: 学習データ中に出現しなかったラベルは無視する
        else:
            used_logit_label_names.append(label)
            used_logit_label_ids.append(ml_label_names.index(label))
    used_logit_label_names = used_logit_label_names
    used_logit_label_ids = np.array(used_logit_label_ids)
    return used_logit_label_names, used_logit_label_ids, category_mapper

def postprocess_for_multi_label_ner_output(snt_len: int, starts: List[str], ends: List[str],
                           outputs, ml_label_names: List[str], positive_cats: List[str], negative_cats: List[str]):
    used_logit_label_names, used_logit_label_ids, category_mapper = load_information_to_extract_focus_and_negative_cats(ml_label_names, positive_cats, negative_cats)
    category_mapped_focus_and_negative_cats = load_category_mapped_focus_and_negative_cats(category_mapper, used_logit_label_names)
    remained_starts, remained_ends, remained_labels, max_probs = extract_focus_and_negative_cats(
        starts, ends, outputs, 
        used_logit_label_names, used_logit_label_ids, category_mapper,
        category_mapped_focus_and_negative_cats, negative_cats
    )
    return load_ner_tags(remained_starts, remained_ends, remained_labels, max_probs, snt_len)

def call_postprocess_for_multi_label_ner_output(example):
    tokens = example["tokens"]
    starts = example["starts"]
    ends = example["ends"]
    outputs = example["outputs"]
    ml_label_names = example["ml_label_names"]
    positive_cats = example["positive_cats"]
    negative_cats: List[str] = example["negative_cats"]
    return postprocess_for_multi_label_ner_output(len(tokens), starts, ends, outputs, ml_label_names, positive_cats, negative_cats)
