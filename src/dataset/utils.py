from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from datasets import DatasetDict
from datasets import load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Callable
from more_itertools import powerset
import os
import re
from inflection import UNCOUNTABLES, PLURALS, SINGULARS
from hydra.utils import get_original_cwd, to_absolute_path

CATEGORY_SEPARATOR = "_"
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


umls_dir = "data/2021AA"
MRSTY = os.path.join(umls_dir, "META", "MRSTY.RRF")
MRCONSO = os.path.join(umls_dir, "META", "MRCONSO.RRF")
SRDEF_PATH = os.path.join(umls_dir, "NET", "SRDEF")


UMLS_ROOT_TUI = "T000"
UMLS_ROOT_SEMANTIC_TYPE = "ROOT"


from anytree import Node


class Node(Node):
    def breadth_first_search(
        self, is_target: Callable[[Node], bool], is_terminal: Callable[[Node], bool]
    ) -> List[Node]:
        search_result = [self] if is_target(self) else []
        if is_terminal(self) or self.is_leaf:
            return search_result
        else:
            for child in self.children:
                search_result += child.breadth_first_search(is_target, is_terminal)
            return search_result


class UMLSNode(Node):
    def __init__(self, tui, semantic_type, parent_node=None, children=None, **kwargs):
        super().__init__(tui, parent_node, children, **kwargs)
        self.tui = tui
        self.semantic_type = semantic_type


def load_umls_thesaurus() -> UMLSNode:
    tui2hier = dict()
    tui2ST = dict()
    with open(to_absolute_path(SRDEF_PATH)) as f:
        for line in f:
            line = line.strip().split("|")
            tui = line[1]
            semantic_type = line[2]
            hier = line[3]
            assert CATEGORY_SEPARATOR not in tui
            if hier.startswith("A") or hier.startswith("B"):
                tui2hier[tui] = hier
                tui2ST[tui] = semantic_type
    parent_tui2children_tui = defaultdict(set)
    hier2tui = {hier: tui for tui, hier in tui2hier.items()}
    for tui in tui2ST.keys():
        child_hier = tui2hier[tui]
        if child_hier in {"A", "B"}:
            parent_tui2children_tui[UMLS_ROOT_TUI].add(tui)
        elif child_hier in {"A1", "A2"}:
            parent_tui2children_tui[hier2tui["A"]].add(tui)
        elif child_hier in {"B1", "B2"}:
            parent_tui2children_tui[hier2tui["B"]].add(tui)
        else:
            parent_hier = ".".join(child_hier.split(".")[:-1])
            parent_tui = hier2tui[parent_hier]
            parent_tui2children_tui[parent_tui].add(tui)
    tui2ST[UMLS_ROOT_TUI] = UMLS_ROOT_SEMANTIC_TYPE

    def make_subtree(root_tui: str, parent_node: UMLSNode):
        subtree_root_node = UMLSNode(root_tui, tui2ST[root_tui], parent_node)
        for child_tui in parent_tui2children_tui[root_tui]:
            make_subtree(child_tui, subtree_root_node)
        return subtree_root_node

    return make_subtree(UMLS_ROOT_TUI, None)


@dataclass
class DatasetConfig:
    name_or_path: str = "conll2003"


tui2hier = dict()
tui2ST = dict()
with open(SRDEF_PATH) as f:
    for line in f:
        line = line.strip().split("|")
        tui = line[1]
        semantic_type = line[2]
        hier = line[3]
        if hier.startswith("A") or hier.startswith("B"):
            tui2hier[tui] = hier
            tui2ST[tui] = semantic_type
STchild2parent = dict()
hier2tui = {hier: tui for tui, hier in tui2hier.items()}
for tui in tui2ST.keys():
    child_hier = tui2hier[tui]
    if child_hier in {"A", "B"}:
        STchild2parent[tui2ST[tui]] = "ROOT"
    elif child_hier in {"A1", "A2"}:
        STchild2parent[tui2ST[tui]] = tui2ST[hier2tui["A"]]
    elif child_hier in {"B1", "B2"}:
        STchild2parent[tui2ST[tui]] = tui2ST[hier2tui["B"]]
    else:
        parent_hier = ".".join(child_hier.split(".")[:-1])
        parent_tui = hier2tui[parent_hier]
        STchild2parent[tui2ST[tui]] = tui2ST[parent_tui]
tui2ST["T000"] = "ROOT"


def get_parent2children():
    STparent2children = defaultdict(list)
    for child, parent in STchild2parent.items():
        STparent2children[parent].append(child)
    return dict(STparent2children)


def get_umls_negative_cats(focus_tuis: List[str]):
    concepts = set(STchild2parent.values())
    tui_recorded_concepts = set(tui2ST.values())
    ST2tui = {SemanticType: tui for tui, SemanticType in tui2ST.items()}
    STparent2children = defaultdict(list)
    for child, parent in STchild2parent.items():
        STparent2children[parent].append(child)
    assert not (concepts - tui_recorded_concepts)
    ascendants_concepts = set()
    focus_STs = {tui2ST[tui] for tui in focus_tuis}
    for focusST in focus_STs:
        candidate_concepts = {focusST}
        while candidate_concepts:
            parent_concept = candidate_concepts.pop()
            if (
                parent_concept in STchild2parent
                and parent_concept not in ascendants_concepts
            ):
                candidate_concepts.add(STchild2parent[parent_concept])
            ascendants_concepts.add(parent_concept)
    ascendants_concepts -= focus_STs
    candidate_negative_STs = set()
    for asc_con in ascendants_concepts:
        candidate_negative_STs |= set(STparent2children[asc_con])
    candidate_negative_STs -= ascendants_concepts
    negative_concepts = candidate_negative_STs - focus_STs
    negative_cats = [ST2tui[concept] for concept in negative_concepts]
    return negative_cats


@cache
def get_tui2ascendants():
    ST2tui = {v: k for k, v in tui2ST.items()}
    tui2ascendants = dict()
    for tui, st in tui2ST.items():
        orig_tui = tui
        ascendants = [tui]
        while tui != "T000":
            st = STchild2parent[st]
            tui = ST2tui[st]
            ascendants.append(tui)
        tui2ascendants[orig_tui] = sorted(ascendants)
    return tui2ascendants


def get_ascendant_tuis(tui: str = "T204") -> List[str]:
    tui2ascendants = get_tui2ascendants()
    return tui2ascendants[tui]


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


def get_umls_negative_cats_from_focus_cats(umls_focus_cat_tuis: List[str]):
    umls_thesaurus = load_umls_thesaurus()
    umls_focus_cat_tuis = set(umls_focus_cat_tuis)

    def is_focus_cats(node: UMLSNode):
        return node.tui in umls_focus_cat_tuis

    def is_negative_cats(node: UMLSNode):
        return not bool(
            set([descendant.tui for descendant in node.descendants])
            & umls_focus_cat_tuis
        ) and not is_focus_cats(node)

    negative_cat_nodes = umls_thesaurus.breadth_first_search(
        is_negative_cats, lambda node: is_focus_cats(node) or is_negative_cats(node)
    )
    umls_negative_cat_tuis = [node.tui for node in negative_cat_nodes]
    umls_negative_cat_tuis.sort()
    assert not umls_focus_cat_tuis & set(umls_negative_cat_tuis)
    return umls_negative_cat_tuis


ST21pvSrc = {
    "CPT",
    "FMA",
    "GO",
    "HGNC",
    "HPO",
    "ICD10",
    "ICD10CM",
    "ICD9CM",
    "MDR",
    "MSH",
    "MTH",
    "NCBI",
    "NCI",
    "NDDF",
    "NDFRT",
    "OMIM",
    "RXNORM",
    "SNOMEDCT_US",
}
