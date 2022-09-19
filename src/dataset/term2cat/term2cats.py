from typing import Dict, List
from seqeval.metrics.sequence_labeling import get_entities
from src.dataset import gold_dataset
from .genia import load_term2cat as genia_load_term2cat
from .twitter import load_twitter_main_dictionary, load_twitter_sibling_dictionary
from hashlib import md5
import os
import json
from dataclasses import dataclass
from omegaconf import MISSING
from .terms import DBPedia_categories, UMLS_Categories
from hydra.utils import get_original_cwd
from collections import defaultdict
import re
from tqdm import tqdm
from src.utils.string_match import ComplexKeywordTyper
from hydra.core.config_store import ConfigStore
from datasets import DatasetDict
from collections import Counter
from prettytable import PrettyTable
from src.dataset.utils import STchild2parent, tui2ST
from .terms import get_ascendants_TUIs


@dataclass
class Term2CatsConfig:
    name: str = MISSING
    output: str = MISSING


@dataclass
class DictTerm2CatsConfig(Term2CatsConfig):
    name: str = "dict"
    focus_cats: str = MISSING
    # duplicate_cats: str = MISSING
    # negative_cats: str = MISSING
    dict_dir: str = os.path.join(os.getcwd(), "data/dict")
    # with_nc: bool = False
    output: str = MISSING
    remove_ambiguate_terms: bool = False


@dataclass
class OracleTerm2CatsConfig(Term2CatsConfig):
    name: str = "oracle"
    gold_dataset: str = MISSING
    output: str = MISSING


def register_term2cat_configs(group="ner_model/typer/term2cat") -> None:
    cs = ConfigStore.instance()
    cs.store(
        group=group,
        name="base_DictTerm2Cats_config",
        node=DictTerm2CatsConfig,
    )
    cs.store(
        group=group,
        name="base_OracleTerm2Cats_config",
        node=OracleTerm2CatsConfig,
    )


def get_anomaly_suffixes(term2cat):
    buffer_file = os.path.join(
        get_original_cwd(),
        "data/buffer/%s"
        % md5(("anomaly_suffixes" + str(term2cat)).encode()).hexdigest(),
    )
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


def log_duplication_between_positive_and_negative_cats(
    cat2terms, positive_cats, negative_cats
):
    positive_terms = set()
    negative_cat2duplicated_terms = defaultdict(set)
    for cat in cat2terms:
        if cat in positive_cats:
            positive_terms |= cat2terms[cat]
    for cat in cat2terms:
        if cat in negative_cats:
            negative_cat2duplicated_terms[cat] |= cat2terms[cat] & positive_terms

    tbl = PrettyTable(["cat", "count", "positive num", "positive ratio"])
    negative_cat2positive_ratio = dict()
    for cat, terms in negative_cat2duplicated_terms.items():
        if len(cat2terms[cat]) > 0:
            positive_ratio = len(terms) / len(cat2terms[cat])
            tbl.add_row([cat, len(cat2terms[cat]), len(terms), positive_ratio])
            negative_cat2positive_ratio[cat] = positive_ratio
    print(tbl.get_string())
    return negative_cat2positive_ratio


def load_dict_term2cats(conf: DictTerm2CatsConfig):
    focus_cats = sorted(set(conf.focus_cats.split("_")))
    term2cats = dict()
    for cat in tqdm(focus_cats):
        print("Start loading: ", cat)
        buffer_file = os.path.join(get_original_cwd(), conf.dict_dir, cat)
        with open(buffer_file) as f:
            # wc = 0
            for line in tqdm(f):
                # for debug
                # wc += 1
                # if wc == 100:
                #     break
                term = line.strip()
                if term:
                    if term not in term2cats:
                        term2cats[term] = cat
                    else:
                        if cat not in term2cats[term]:
                            term2cats[term] += "_%s" % cat
    # 階層構造的にあり得るカテゴリの組み合わせを列挙する
    print(Counter(term2cats.values()).most_common())
    print("term2cats length: ", len(term2cats))
    if conf.remove_ambiguate_terms:
        acceptable_cat_combination = {
            "_".join(sorted(get_ascendants_TUIs(tui))) for tui in focus_cats
        }
        remove_terms = []
        for term, cats in term2cats.items():
            if cats not in acceptable_cat_combination:
                remove_terms.append(term)
        for term in remove_terms:
            del term2cats[term]
        print("term2cats length after removing ambiguate terms: ", len(term2cats))
        print(Counter(term2cats.values()).most_common())
    return term2cats


def load_oracle_term2cat(conf: OracleTerm2CatsConfig):
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


def load_term2cats(conf: Term2CatsConfig):
    if conf.name == "dict":
        term2cat = load_dict_term2cats(conf)
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
    args = str(with_sibilling) + str(sibling_compression) + str(only_fake)
    buffer_file = "data/buffer/%s" % md5(args.encode()).hexdigest()
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
        args = " ".join(
            map(str, [task, with_sibling, sibilling_compression, only_fake])
        )
        buffer_file = os.path.join("data/buffer", md5(args.encode()).hexdigest())
        if task == "JNLPBA":
            term2cat = genia_load_term2cat(
                with_sibling, sibilling_compression, only_fake
            )
        elif task == "Twitter":
            term2cat = load_twitter_dictionary(
                with_sibling, sibilling_compression, only_fake
            )
        self.term2cat = term2cat


def log_term2cats(term2cats: Dict):
    print("log term2cat count")
    tbl = PrettyTable(["cats", "count"])
    counter = Counter(term2cats.values())
    for cats, count in sorted(list(counter.items()), key=lambda x: x[0]):
        tbl.add_row([cats, count])
    print(tbl.get_string())
    print("category num: ", len(counter))


    counter = Counter(term2cats.values())
    cat2count = defaultdict(lambda: 0)
    for cats, count in sorted(list(counter.items()), key=lambda x: x[0]):
        for cat in cats.split('_'):
            cat2count[cat] += count

    tbl = PrettyTable(["cat", "Semantic Type", "count"])
    for cat, count in cat2count.items():
        tbl.add_row([cat, tui2ST[cat], count])
    print(tbl.get_string())
    print("category num: ", len(counter))
