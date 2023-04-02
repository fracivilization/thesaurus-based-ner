from typing import Dict, List
from seqeval.metrics.sequence_labeling import get_entities
from .genia import load_term2cat as genia_load_term2cat
from .twitter import load_twitter_main_dictionary, load_twitter_sibling_dictionary
import os
from dataclasses import dataclass
from omegaconf import MISSING
from hydra.utils import get_original_cwd, to_absolute_path
from collections import defaultdict
from src.utils.string_match import ComplexKeywordTyper
from hydra.core.config_store import ConfigStore
from datasets import DatasetDict
from collections import Counter
from prettytable import PrettyTable
from src.utils.utils import WeightedSQliteDict
from src.dataset.utils import load_dbpedia_thesaurus, get_negative_cats_from_focus_cats
from tqdm import tqdm


@dataclass
class Term2CatConfig:
    term2cats: str = MISSING
    name: str = MISSING
    output: str = MISSING


@dataclass
class DictTerm2CatConfig(Term2CatConfig):
    name: str = "dict"
    focus_cats: str = MISSING
    # duplicate_cats: str = MISSING
    negative_cats: str = MISSING
    term2cats: str = MISSING
    # with_nc: bool = False
    remove_anomaly_suffix: bool = False  # remove suffix term (e.g. "migration": nc-T054 for "cell migration": T038)
    output: str = MISSING


@dataclass
class OracleTerm2CatConfig(Term2CatConfig):
    name: str = "oracle"
    gold_dataset: str = MISSING
    output: str = MISSING


def register_term2cat_configs(group="ner_model/typer/term2cat") -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=group,
        name="base_DictTerm2Cat_config",
        node=DictTerm2CatConfig,
    )
    cs.store(
        group=group,
        name="base_OracleTerm2Cat_config",
        node=OracleTerm2CatConfig,
    )


def get_anomaly_suffixes(term2cat):
    anomaly_suffixes = set()
    complex_typer = ComplexKeywordTyper(term2cat)
    lowered2orig = defaultdict(list)
    for term in term2cat:
        lowered2orig[term.lower()].append(term)
    for term, cat in term2cat.items():
        confirmed_common_suffixes = complex_typer.get_confirmed_common_suffixes(term)
        for pred_cat, start in confirmed_common_suffixes:
            if pred_cat != cat and start != 0:
                anomaly_suffix = term[start:]
                lowered2orig[anomaly_suffix]
                for orig_term in lowered2orig[anomaly_suffix]:
                    anomaly_suffixes.add(orig_term)
    return anomaly_suffixes


CategoryMapper = {
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
        # NOTE:ethnicGroupは'/<http://www.w3.org/2002/07/owl#Class>/<http://www.w3.org/1999/02/22-rdf-syntax-ns#Property>/<http://dbpedia.org/ontology/ethnicGroup>' という箇所にある
        "<http://dbpedia.org/ontology/ethnicGroup>",
    },
}


def get_dbpedia_negative_cats_from_focus_cats(focus_cats: List[str]):
    dbpedia_thesaurus = load_dbpedia_thesaurus()

    negative_cats = get_negative_cats_from_focus_cats(focus_cats, dbpedia_thesaurus)
    assert not focus_cats & set(negative_cats)
    return negative_cats


def load_dict_term2cat(conf: DictTerm2CatConfig):
    focus_cats = set(conf.focus_cats.split("_"))
    if conf.negative_cats:
        negative_cats = set(conf.negative_cats.split("_"))
    else:
        negative_cats = set()
    target_cats = focus_cats | negative_cats
    # NOTE: PER, LOC, ORG, MISCなどKnowledgebaseと対応付ける必要があるカテゴリに対処する
    target_knowledge_base_cats = set()
    for target_cat in target_cats:
        if target_cat in CategoryMapper:
            for knowledge_base_cat in CategoryMapper[target_cat]:
                target_knowledge_base_cats.add(knowledge_base_cat)
        else:
            target_knowledge_base_cats.add(target_cat)
    knowledgebase_cat2target_cat = {
        kb_cat: cat
        for cat, knowledgebase_cats in CategoryMapper.items()
        for kb_cat in knowledgebase_cats
    }
    term2cats = WeightedSQliteDict.load_from_disk(to_absolute_path(conf.term2cats))

    term2cat = dict()
    total_count = len(term2cats)
    for term, weighted_cats in tqdm(term2cats.items(), total=total_count):
        candidate_knowledgebase_cats = (
            set(weighted_cats.values) & target_knowledge_base_cats
        )
        candidate_cats = set()
        for kb_cat in candidate_knowledgebase_cats:
            if kb_cat in knowledgebase_cat2target_cat:
                candidate_cats.add(knowledgebase_cat2target_cat[kb_cat])
            else:
                candidate_cats.add(kb_cat)

        if len(candidate_cats) == 1:
            knowledgebase_cat = candidate_cats.pop()
            cat = (
                knowledgebase_cat2target_cat[knowledgebase_cat]
                if knowledgebase_cat in knowledgebase_cat2target_cat
                else knowledgebase_cat
            )
        elif len(candidate_cats) > 1:
            kb_cat2weight = {
                cat: weight
                for cat, weight in zip(weighted_cats.values, weighted_cats.weights)
            }
            candidate_cat2weight = defaultdict(lambda: 0)
            for candidate_cat in candidate_cats:
                if candidate_cat in CategoryMapper:
                    for correspond_kb_cat in CategoryMapper[candidate_cat]:
                        if correspond_kb_cat in kb_cat2weight:
                            candidate_cat2weight[candidate_cat] = max(
                                kb_cat2weight[correspond_kb_cat],
                                candidate_cat2weight[candidate_cat],
                            )
                else:
                    candidate_cat2weight[candidate_cat] = max(
                        kb_cat2weight[candidate_cat],
                        candidate_cat2weight[candidate_cat],
                    )
            highest_weight = max(candidate_cat2weight.values())
            highest_weighted_cats = [
                cat
                for cat, weight in candidate_cat2weight.items()
                if weight == highest_weight
            ]
            if len(highest_weighted_cats) == 1:
                cat = highest_weighted_cats.pop()
            else:
                continue
        else:
            continue

        if cat in negative_cats:
            term2cat[term] = "nc-%s" % cat
        else:
            term2cat[term] = cat

    if conf.remove_anomaly_suffix:
        anomaly_suffixes = get_anomaly_suffixes(term2cat)
        for term in anomaly_suffixes:
            del term2cat[term]
    return term2cat


def load_oracle_term2cat(conf: OracleTerm2CatConfig):
    gold_datasets = DatasetDict.load_from_disk(
        os.path.join(get_original_cwd(), conf.gold_dataset)
    )
    cat2terms = defaultdict(set)
    for key, split in gold_datasets.items():
        label_names = split.features["ner_tags"].feature.names
        for snt in split:
            for cat, s, e in get_entities(
                [label_names[tag] for tag in snt["ner_tags"]]
            ):
                term = " ".join(snt["tokens"][s : e + 1])
                cat2terms[cat].add(term)
    remove_terms = set()
    for i1, (c1, t1) in enumerate(cat2terms.items()):
        for i2, (c2, t2) in enumerate(cat2terms.items()):
            if i2 > i1:
                duplicated = t1 & t2
                if duplicated:
                    remove_terms |= duplicated
                    # for t in duplicated:
                    # term2cats[t] |= {c1, c2}
    term2cat = dict()
    for cat, terms in cat2terms.items():
        for non_duplicated_term in terms - remove_terms:
            term2cat[non_duplicated_term] = cat
    return term2cat


def load_term2cat(conf: Term2CatConfig):
    if conf.name == "dict":
        term2cat = load_dict_term2cat(conf)
    elif conf.name == "oracle":
        term2cat = load_oracle_term2cat(conf)
    else:
        raise NotImplementedError
    return term2cat


def load_jnlpba_main_term2cat():
    pass


def load_jnlpba_dictionary(
    with_sibilling: bool = False,
    sibilling_compression: str = "none",
    only_fake: bool = False,
):
    term2cat = load_jnlpba_main_term2cat()
    if with_sibilling:
        raise NotImplementedError
    return term2cat


def load_twitter_dictionary(
    with_sibilling: bool = True,
    sibling_compression: str = "none",
    only_fake: bool = True,
):
    term2cat = dict()
    main_dictionary = load_twitter_main_dictionary()
    term2cat.update({k: v for k, v in main_dictionary.items() if v != "product"})
    if with_sibilling:
        sibling_dict = load_twitter_sibling_dictionary(sibling_compression)
        for k, v in sibling_dict.items():
            if k not in term2cat:
                term2cat[k] = v
    if only_fake:
        term2cat = {k: v for k, v in term2cat.items() if v.startswith("fake_")}
    else:
        for k, v in main_dictionary.items():
            if v == "product" and k not in term2cat:
                term2cat[k] = "product"
    return term2cat


class Term2Cat:
    def __init__(
        self,
        task: str,
        with_sibling: bool = False,
        sibilling_compression: str = "none",
        only_fake: bool = False,
    ) -> None:
        assert sibilling_compression in {"all", "sibilling", "none"}
        if task == "JNLPBA":
            term2cat = genia_load_term2cat(
                with_sibling, sibilling_compression, only_fake
            )
        elif task == "Twitter":
            term2cat = load_twitter_dictionary(
                with_sibling, sibilling_compression, only_fake
            )
        self.term2cat = term2cat


def log_term2cat(term2cat: Dict):
    print("log term2cat count")
    tbl = PrettyTable(["cat", "count"])
    counter = Counter(term2cat.values())
    for cat, count in sorted(list(counter.items()), key=lambda x: x[0]):
        tbl.add_row([cat, count])
    print(tbl.get_string())
    print("category num: ", len(counter))
