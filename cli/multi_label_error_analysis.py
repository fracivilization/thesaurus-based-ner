import random
from datasets import Dataset, DatasetDict
import numpy as np
from scipy.special import softmax
from seqeval.metrics.sequence_labeling import get_entities
from src.utils.tree_visualize import Node, get_tree_str_for_current_node, tree_repr
from src.dataset.utils import get_parent2children, tui2ST
from typing import List
from more_itertools import powerset


def valid_label_set():
    parent2children = get_parent2children()
    parent2children["UMLS"] = parent2children["ROOT"]
    parent2children["ROOT"] = ["UMLS", "nc-O"]
    ST2tui = {val: key for key, val in tui2ST.items()}
    ST2tui["nc-O"] = "nc-O"
    child2parent = {
        child: parent
        for parent, children in parent2children.items()
        for child in children
    }
    label_names = list(child2parent.keys())

    def ascendant_labels(child_label) -> List[str]:
        ascendants = [child_label]
        while child_label != "ROOT":
            parent = child2parent[child_label]
            ascendants.append(parent)
            child_label = parent
        pass
        return ascendants

    valid_sets = set()
    valid_paths = set(["T000"])
    label2valid_paths = {"T000": "T000"}
    for label in label_names:
        if label in {"UMLS", "ROOT"}:
            continue
        valid_labels = ascendant_labels(label)
        if "UMLS" in valid_labels:
            valid_labels.remove("UMLS")
            valid_labels.append("ROOT")
        if "ROOT" in valid_labels:
            valid_labels.remove("ROOT")
        valid_labels = [ST2tui[vl] for vl in valid_labels]
        valid_path = "_".join(sorted(valid_labels))
        valid_paths.add(valid_path)
        label2valid_paths[ST2tui[label]] = valid_path
        valid_sets |= set(
            ["_".join(sorted(labels)) for labels in powerset(valid_labels) if labels]
        )
    return valid_sets, valid_paths, label2valid_paths


valid_label_set, valid_paths, label2valid_paths = valid_label_set()
label2depth = {label: len(path.split("_")) for label, path in label2valid_paths.items()}


def hierarchical_valid(labels: List[str]):
    """入力されたラベル集合が階層構造として妥当であるかを判断する。

    ここで、階層構造として妥当であるとは、そのラベル集合が("O"ラベルを含んだシソーラスにおける)
    ルートノードから各ノードへのパス、あるいはその部分集合となるようなラベル集合のことである

    Args:
        labels (List[str]): 妥当性を判断するラベル集合
    """
    return "_".join(sorted(labels)) in valid_label_set


def get_complete_path(labels: List[str]):
    # 最も深いレベルに存在するラベルを取得する
    depths = [label2depth[l] for l in labels]
    # そのラベルへのパスとなるラベル集合を取得する
    return label2valid_paths[labels[depths.index(max(depths))]].split("_")


def ranked_label2hierarchical_valid_labels(ranked_labels: List[str]):
    """ランク付けされたラベルのリストから階層構造的に曖昧性のなく妥当なラベルセットを出力する

    Args:
        ranked_labels (List[str]): ラベルが出力されたランク順に並んだリスト
    """
    hierarchical_valid_labels = []
    for label in ranked_labels:
        if not hierarchical_valid(hierarchical_valid_labels + [label]):
            break
        else:
            hierarchical_valid_labels.append(label)
    if "_".join(sorted(hierarchical_valid_labels)) not in valid_paths:
        hierarchical_valid_labels = get_complete_path(hierarchical_valid_labels)
    return hierarchical_valid_labels


def get_tree_from_logits(logits, label_names):
    parent2children = get_parent2children()
    parent2children["UMLS"] = parent2children["ROOT"]
    parent2children["ROOT"] = ["UMLS", "nc-O"]
    label_names = [tui2ST[label] if label in tui2ST else label for label in label_names]
    label_names[label_names.index("ROOT")] = "UMLS"
    parent2children_dict = parent2children
    "# 注TOOOとnc-OをROOTにつなげる"
    node2prob = dict()
    ROOT = Node("ROOT")
    probs = softmax(logits)

    def descendant_nodes(parent_node=ROOT, parent_name="ROOT"):
        if parent_name in parent2children_dict:
            for child_name in parent2children_dict[parent_name]:
                prob = probs[label_names.index(child_name)]
                child_node = Node("%.2f %s" % (100 * prob, child_name), parent_node)
                descendant_nodes(child_node, child_name)

    descendant_nodes()
    return get_tree_str_for_current_node(ROOT)
    # return tree_repr(ROOT)
    pass


FOCUS_CATS = "T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204".split()
NEGATIVE_CATS = "T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200".split()

d_multi = Dataset.load_from_disk(
    "outputs/2022-04-10/22-57-17/span_classif_log_2c63ad62-b8dc-11ec-8602-ac1f6b017a9e"
)
d_single = Dataset.load_from_disk(
    "outputs/2022-04-11/22-55-02/span_classif_log_e047532e-b99f-11ec-8602-ac1f6b017a9e"
)
d_gold = DatasetDict.load_from_disk(
    "data/gold/836053d1bd47c7cf824672a17714a61a354af8e9"
)["validation"]
# 正解以外の部分にどれくらい確率が付与されるかを確認する

for snt_multi, snt_single, snt_gold in zip(d_multi, d_single, d_gold):
    label_names_multi = snt_multi["label_names"]
    label_names_single = snt_single["label_names"]
    label_names_gold = d_gold.features["ner_tags"].feature.names
    gold_tags = [label_names_gold[tag] for tag in snt_gold["ner_tags"]]
    tokens = snt_multi["tokens"]
    gold_mentions = [
        (" ".join(tokens[s : e + 1]), l) for l, s, e in get_entities(gold_tags)
    ]

    logits_multi = np.array(snt_multi["logits"])
    logits_single = np.array(snt_single["logits"])

    focus_probs_multi = softmax(
        np.array(
            [
                logits_multi[:, label_names_multi.index(label)]
                for label in ["nc-O"] + FOCUS_CATS + NEGATIVE_CATS
            ]
        ).T,
        axis=1,
    )
    focus_probs_single = softmax(
        np.array(
            [
                logits_single[:, label_names_single.index(label)]
                for label in ["nc-O"] + FOCUS_CATS
            ]
        ).T,
        axis=1,
    )
    assert snt_multi["starts"] == snt_single["starts"]
    assert snt_multi["ends"] == snt_single["ends"]
    print("sentence: ", " ".join(tokens))
    print("gold NEs: ", gold_mentions)
    multi_predictions = []
    single_predictions = []
    for prob_multi, logit_multi, prob_single, start, end in zip(
        focus_probs_multi,
        logits_multi,
        focus_probs_single,
        snt_multi["starts"],
        snt_multi["ends"],
    ):
        mention = " ".join(tokens[start:end])
        multi_max_likelihood = prob_multi.max()
        ranked_label = [(label_names_multi)[i] for i in (-logit_multi).argsort()]
        valid_labels = ranked_label2hierarchical_valid_labels(ranked_label)
        # pred_multi = (["nc-O"] + FOCUS_CATS + NEGATIVE_CATS)[prob_multi.argmax()]
        focus_labels = set(valid_labels) & set(FOCUS_CATS)
        if focus_labels:
            pred_multi = random.choice(list(focus_labels))
        else:
            pred_multi = "nc-O"
        single_max_likelihood = prob_single.max()
        pred_single = (["nc-O"] + FOCUS_CATS)[prob_single.argmax()]
        # print("multi", pred_multi, multi_max_likelihood)
        # print("single", pred_single, single_max_likelihood)
        if pred_multi in FOCUS_CATS:
            multi_predictions.append(
                (
                    mention,
                    tui2ST[pred_multi],
                    "%.2f" % (100 * multi_max_likelihood,),
                    # get_tree_from_logits(logit_multi, label_names_multi),
                )
            )
        if not pred_single.startswith("nc-"):
            single_predictions.append(
                (mention, tui2ST[pred_single], "%.2f" % (100 * single_max_likelihood,))
            )
    multi_predictions = sorted(multi_predictions, key=lambda x: x[2], reverse=True)
    single_predictions = sorted(single_predictions, key=lambda x: x[2], reverse=True)
    print("\n\n")
    # print("multi: ")
    # for mult_pred in multi_predictions:
    #     print(mult_pred[:3])
    #     print(mult_pred[3])
    print("multi: ", multi_predictions)
    # print("\n\n")
    print("single: ", single_predictions)
    pass
