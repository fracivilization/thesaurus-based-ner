import os
import pickle
from typing import Dict, Set, List
from tqdm import tqdm
from collections import defaultdict
from rdflib import Graph, URIRef
import re
from logging import getLogger
from hydra.utils import to_absolute_path
from src.dataset.utils import (
    ST21pvSrc,
    tui2ST,
    load_dbpedia_parent2descendants,
    get_dbpedia_negative_cats_from_positive_cats,
    load_DBPediaCategories,
)

logger = getLogger(__name__)

umls_dir = "data/2021AA"
mrsty = os.path.join(umls_dir, "META", "MRSTY.RRF")
mrconso = os.path.join(umls_dir, "META", "MRCONSO.RRF")
srdef = os.path.join(umls_dir, "NET", "SRDEF")
DBPedia_dir = "data/DBPedia"


def get_descendants_TUIs(tui="T025"):
    tui2stn: Dict[str, str] = dict()
    with open(srdef) as f:
        for line in f:
            line = line.strip().split("|")
            if line[1] not in tui2stn:
                tui2stn[line[1]] = line[3]
            else:
                raise NotImplementedError
    if tui == "T000":
        entities = {tui for tui, stn in tui2stn.items() if stn.startswith("A")}
        events = {tui for tui, stn in tui2stn.items() if stn.startswith("B")}
        descendants_tuis = entities | events
    else:
        root_stn = tui2stn[tui]
        descendants_tuis = {
            tui for tui, stn in tui2stn.items() if stn.startswith(root_stn)
        }
    return descendants_tuis


def load_TUI_terms(tui="T025"):
    include_tuis = get_descendants_TUIs(tui)
    with open(mrsty) as f:
        cuis = [
            line.strip().split("|")[0]
            for line in f
            if line.split("|")[1] in include_tuis
        ]
    cuis = set(cuis)
    cui2terms = defaultdict(set)
    terms = set()
    with open(mrconso) as f:
        for line in tqdm(f, total=16132274):
            (
                cui,
                lang,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                src,
                _,
                _,
                term,
                _,
                _,
                _,
                _,
            ) = line.strip().split("|")
            if lang == "ENG" and src in ST21pvSrc and cui in cuis:
                terms.add(term)
    return terms


def load_tui2cui_count():
    tui2cuis = defaultdict(set)
    with open(mrsty) as f:
        for line in f:
            line = line.strip().split("|")
            tui2cuis[line[1]].add(line[0])
        tui2cui_count = {tui: len(cuis) for tui, cuis in tui2cuis.items()}
    return tui2cui_count


CoNLL2003Categories = {"PER", "ORG", "LOC", "MISC"}


DBPediaCategories = load_DBPediaCategories()


CoNLL2003ToDBPediaCategoryMapper = {
    # NOTE: MISCはこれらいずれにも属さないカテゴリとする
    "PER": {"Person"},
    "ORG": {"Organisation"},
    "LOC": {"Place"},
}
