from collections import defaultdict
from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Set
from more_itertools import powerset
import os
import re
from inflection import UNCOUNTABLES, PLURALS, SINGULARS
from hydra.utils import to_absolute_path
from functools import lru_cache
from datasets import Dataset
import random

CATEGORY_SEPARATOR = "_"
NEGATIVE_CATEGORY_PREFIX = "nc"
NEGATIVE_CATEGORY_TEMPLATE = f"{NEGATIVE_CATEGORY_PREFIX}-%s"
PLURAL_RULES = [(re.compile(rule), replacement) for rule, replacement in PLURALS]
SINGULAR_RULES = [(re.compile(rule), replacement) for rule, replacement in SINGULARS]
dbpedia_ontology_pattern = re.compile("<http://dbpedia.org/ontology/([^>]+)>")

CoNLL2003CategoryMapper = {
    # NOTE: MISCはこれらいずれにも属さないカテゴリとする
    "PER": {
        "<http://dbpedia.org/ontology/Person>",
        "<http://dbpedia.org/ontology/Name>",
    },
    "ORG": {"<http://dbpedia.org/ontology/Organisation>"},
    "LOC": {"<http://dbpedia.org/ontology/Place>"},
    # NOTE: DBPediaにはtime, day, ballなどの大量の一般名詞がふくまれるので、
    #       PER, ORG, LOC以外とするのではなく、結局列挙する必要がありそう
    "MISC": {
        "<http://dbpedia.org/ontology/Work>",
        "<http://dbpedia.org/ontology/Event>",
        "<http://dbpedia.org/ontology/MeanOfTransportation>",
        "<http://dbpedia.org/ontology/Device>",  # AK-47が含まれているので
        "<http://dbpedia.org/ontology/Award>",  # ノーベル平和賞がふくまれているので
        "<http://dbpedia.org/ontology/Disease>",
        # # NOTE:ethnicGroupは'/<http://www.w3.org/2002/07/owl#Class>/<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>/<http://dbpedia.org/ontology/ethnicGroup>' という箇所にある
        # "<http://dbpedia.org/ontology/ethnicGroup>",
        # NOTE: 属性としてのethnicGroupと名称としてのEthnicGroupの両方があるらしい
        "<http://dbpedia.org/ontology/EthnicGroup>",
    },
}
CoNLL2003CommonNounMapper = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "MISC": "miscellaneous",
}


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


def dbpedia_name_patch(node="<http://dbpedia.org/ontology/Infrastucture>"):
    dbpedia_name_patch = {
        "<http://dbpedia.org/ontology/Infrastucture>": "<http://dbpedia.org/ontology/Infrastructure>",
        "<http://dbpedia.org/ontology/TimeInterval>": "<http://dbpedia.org/ontology/TimePeriod>",
    }
    if node in dbpedia_name_patch:
        return dbpedia_name_patch[node]
    else:
        return node


def load_dbpedia_parent2descendants() -> Dict:
    DBPedia_dir = "data/DBPedia"
    DBPedia_ontology = os.path.join(DBPedia_dir, "ontology--DEV_type=parsed_sorted.nt")
    # Get DBPedia Concepts
    from rdflib import Graph, URIRef

    g = Graph()
    g.parse(to_absolute_path(DBPedia_ontology))
    s, p, o = next(g.__iter__())
    parent2children = defaultdict(set)
    for s, p, o in g:
        if isinstance(o, URIRef) and p.n3() in {
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
            "<http://www.w3.org/2000/01/rdf-schema#subClassOf>",
        }:
            parent2children[dbpedia_name_patch(o.n3())] |= {dbpedia_name_patch(s.n3())}
    return parent2children


@lru_cache(maxsize=None)
def load_dbpedia_thesaurus() -> Node:
    parent2children = load_dbpedia_parent2descendants()
    import networkx as nx

    parent_child_pair = [
        (parent, child)
        for parent, children in parent2children.items()
        for child in children
    ]
    graph = nx.DiGraph(parent_child_pair)
    reducted_graph = nx.transitive_reduction(graph)
    # NOTE: ルートノードの候補として"<http://www.w3.org/2002/07/owl#Class>" と　"<http://www.w3.org/2002/07/owl#Thing>" がある
    # NOTE: 前者にはElectricalSubstation, Holidayが余計に含まれている
    # NOTE: そこでowl#Classを基準に利用しつつ、
    # NOTE: もしowl#Class, owl#Thing以外の親クラスを持つ場合にはその親クラスに置き換えるという処理を行う。
    candidate_classes = set(
        list(reducted_graph.successors("<http://www.w3.org/2002/07/owl#Class>"))
    )
    top_classses = set()
    root_classes = {
        "<http://www.w3.org/2002/07/owl#Class>",
        "<http://www.w3.org/2002/07/owl#Thing>",
        # NOTE: 次のものも不要なので排除
        "<http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#TimeInterval>",
    }
    while candidate_classes:
        candidates = candidate_classes.copy()
        for candidate in candidates:
            candidate_ancestors = nx.ancestors(reducted_graph, candidate)
            candidate_ancestors -= root_classes
            if candidate_ancestors:
                candidate_ancestors -= top_classses
                candidate_classes |= candidate_ancestors
            else:
                top_classses.add(candidate)
            candidate_classes.remove(candidate)
    parent2children["<http://www.w3.org/2002/07/owl#Class>"] = top_classses
    parent2children["<http://www.w3.org/2002/07/owl#Class>"] = top_classses

    def make_subtree(node_name: str, parent_node: UMLSNode):
        subtree_root_node = Node(node_name, parent_node)
        for child_tui in parent2children[node_name]:
            make_subtree(child_tui, subtree_root_node)
        return subtree_root_node

    return make_subtree("<http://www.w3.org/2002/07/owl#Class>", None)


@lru_cache(maxsize=None)
def load_dbpedia_thesaurus_name2node() -> Dict[str, Node]:
    dbpedia_thesaurus = load_dbpedia_thesaurus()
    name2node = {node.name: node for node in dbpedia_thesaurus.descendants}
    name2node[dbpedia_thesaurus.name] = dbpedia_thesaurus
    name2node["<http://www.w3.org/2002/07/owl#Thing>"] = dbpedia_thesaurus
    return name2node


def get_ascendant_dbpedia_thesaurus_node(node_name: str) -> List[str]:
    name2node = load_dbpedia_thesaurus_name2node()
    if node_name not in name2node:
        return []
    ascendants = [node.name for node in name2node[node_name].ancestors]
    # 自分自身を追加する。ただし、owl#Thing は owl#Classと同じとみなす
    return ascendants + [name2node[node_name].name]


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


@lru_cache(maxsize=None)
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


def get_negative_cats_from_positive_cats(
    positive_cats: Set[str], root_node_of_thesaurus: Node
):
    def is_positive_cats(node: Node):
        return node.name in positive_cats

    def is_negative_cats(node: Node):
        return not bool(
            set([descendant.name for descendant in node.descendants]) & positive_cats
        ) and not is_positive_cats(node)

    negative_cat_nodes = root_node_of_thesaurus.breadth_first_search(
        is_negative_cats, lambda node: is_positive_cats(node) or is_negative_cats(node)
    )
    negative_cat_names = [node.name for node in negative_cat_nodes]

    # NOTE: DBPediaの場合のみURL部分を取り除く
    if root_node_of_thesaurus.name == "T000":
        pass
    elif root_node_of_thesaurus.name == "<http://www.w3.org/2002/07/owl#Class>":
        for negative_cat_name in negative_cat_names:
            assert dbpedia_ontology_pattern.match(
                negative_cat_name
            ) or negative_cat_name in {
                "<http://www.w3.org/2002/07/owl#ObjectProperty>",
                "<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>",
                "<http://www.w3.org/2002/07/owl#DatatypeProperty>",
            }
    else:
        raise NotImplementedError
    negative_cat_names.sort()
    return negative_cat_names


def load_negative_cats_from_positive_cats(
    positive_categories: List[str], eval_dataset: str
):
    if eval_dataset == "MedMentions":
        negative_cats = get_umls_negative_cats_from_positive_cats(
            positive_categories
        )
    elif eval_dataset == "CoNLL2003":
        negative_cats = get_dbpedia_negative_cats_from_positive_cats(
            positive_categories
        )
    return negative_cats


def get_umls_negative_cats_from_positive_cats(positive_cats: List[str]):
    thesaurus = load_umls_thesaurus()
    positive_cats = set(positive_cats)

    negative_cats = get_negative_cats_from_positive_cats(positive_cats, thesaurus)
    assert not positive_cats & set(negative_cats)
    return negative_cats


def get_dbpedia_negative_cats_from_positive_cats(positive_cats: List[str]):
    dbpedia_thesaurus = load_dbpedia_thesaurus()
    new_positive_cats = []
    for positive_cat in positive_cats:
        if positive_cat in CoNLL2003CategoryMapper:
            new_positive_cats.extend(CoNLL2003CategoryMapper[positive_cat])
        else:
            new_positive_cats.append(positive_cat)
    positive_cats = set(new_positive_cats)

    negative_cats = get_negative_cats_from_positive_cats(
        positive_cats, dbpedia_thesaurus
    )
    assert not positive_cats & set(negative_cats)
    return negative_cats


def load_DBPediaCategories():
    dbpedia_thesaurus = load_dbpedia_thesaurus()
    categories = []
    for node in dbpedia_thesaurus.descendants:
        matched = dbpedia_ontology_pattern.match(node.name)
        if matched:
            categories.append(matched.group(1))
    categories.sort()
    return categories


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


def split_ner_dataset(dataset: Dataset, split_num: int = 10) -> List[Dataset]:
    """指定した数に固有表現抽出データセットを分割する"""
    dataset_dicts = [defaultdict(list) for i in range(split_num)]
    for snt in dataset:
        target_split_data_id = random.randrange(split_num)
        for key, value in snt.items():
            dataset_dicts[target_split_data_id][key].append(value)
    return [
        Dataset.from_dict(split, features=dataset.features) for split in dataset_dicts
    ]


def translate_label_name_into_common_noun(label_name: str) -> str:
    """labelに使用するタグをSDNetで利用する一般名詞に変換する
    e.g.
    - Input: LOC
    - Output: location
    """
    if label_name in CoNLL2003CommonNounMapper:
        return CoNLL2003CommonNounMapper[label_name]
    elif label_name in tui2ST:
        return tui2ST[label_name].lower()
    else:
        raise NotImplementedError
