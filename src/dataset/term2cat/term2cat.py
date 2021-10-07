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


@dataclass
class Term2CatConfig:
    focus_cats: str = MISSING
    duplicate_cats: str = MISSING
    dict_dir: str = os.path.join(os.getcwd(), "data/dict")


def load_term2cat(conf: Term2CatConfig):
    focus_cats = set(conf.focus_cats.split("_"))
    duplicate_cats = set(conf.duplicate_cats.split("_")) | focus_cats
    remained_nc = DBPedia_categories | UMLS_Categories
    remained_nc = remained_nc - duplicate_cats  # nc: negative cat
    # remained_nc = set()
    cats = focus_cats | remained_nc
    cat2terms = dict()
    for cat in cats:
        with open(os.path.join(conf.dict_dir, cat)) as f:
            pass
            terms = f.read().split("\n")
            cat2terms[cat] = set(terms)
    duplicate_terms = set()
    # term2cats = defaultdict(set)
    for i1, (c1, t1) in enumerate(cat2terms.items()):
        for i2, (c2, t2) in enumerate(cat2terms.items()):
            if i2 > i1:
                duplicated = t1 & t2
                if duplicated:
                    duplicate_terms |= duplicated
                    # for t in duplicated:
                    # term2cats[t] |= {c1, c2}
    term2cat = dict()
    for cat, terms in cat2terms.items():
        for non_duplicated_term in terms - duplicate_terms:
            if cat in remained_nc:
                term2cat[non_duplicated_term] = "nc-%s" % cat
            else:
                term2cat[non_duplicated_term] = cat
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
    if not os.path.exists(buffer_file):
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
        with open(buffer_file, "w") as f:
            json.dump(term2cat, f)
    with open(buffer_file) as f:
        term2cat = json.load(f)
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
        if not os.path.exists(buffer_file):
            if task == "JNLPBA":
                term2cat = genia_load_term2cat(
                    with_sibling, sibilling_compression, only_fake
                )
            elif task == "Twitter":
                term2cat = load_twitter_dictionary(
                    with_sibling, sibilling_compression, only_fake
                )
            pass
            with open(buffer_file, "w") as f:
                json.dump(term2cat, f)
        with open(buffer_file, "r") as f:
            term2cat = json.load(f)
        self.term2cat = term2cat
