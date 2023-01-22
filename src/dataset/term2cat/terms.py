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
    get_dbpedia_negative_cats_from_focus_cats,
    load_DBPediaCategories,
)

logger = getLogger(__name__)

umls_dir = "data/2021AA"
mrsty = os.path.join(umls_dir, "META", "MRSTY.RRF")
mrconso = os.path.join(umls_dir, "META", "MRCONSO.RRF")
srdef = os.path.join(umls_dir, "NET", "SRDEF")
DBPedia_dir = "data/DBPedia"
# DBPedia(Wikipedia)
DBPedia_ontology = os.path.join(DBPedia_dir, "ontology--DEV_type=parsed_sorted.nt")
DBPedia_instance_type = os.path.join(DBPedia_dir, "instance-types_lang=en_specific.ttl")
DBPedia_mapping_literals = os.path.join(
    DBPedia_dir, "mappingbased-literals_lang=en.ttl"
)
DBPedia_infobox = os.path.join(DBPedia_dir, "infobox-properties_lang=en.ttl")
DBPedia_redirect = os.path.join(DBPedia_dir, "redirects_lang=en.ttl")
# DBPedia (Wikidata)
DBPedia_WD_instance_type = os.path.join(DBPedia_dir, "instance-types_specific.ttl")
DBPedia_WD_SubClassOf = os.path.join(DBPedia_dir, "ontology-subclassof.ttl")
DBPedia_WD_labels = os.path.join(DBPedia_dir, "labels.ttl")
DBPedia_WD_alias = os.path.join(DBPedia_dir, "alias.ttl")


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


def get_article2names():
    import pickle

    output = os.path.join(DBPedia_dir, "article2names.pkl")
    article2names = defaultdict(set)
    label_predicates = {
        "<http://www.w3.org/2000/01/rdf-schema#label>",
        "<http://xmlns.com/foaf/0.1/name>",
        "<http://dbpedia.org/property/name>",
        "<http://dbpedia.org/property/alias>",
        "<http://dbpedia.org/ontology/alias>",
    }
    pattern = "(<[^>]+>) " + "(%s)" % "|".join(label_predicates) + ' "([^"]+)"@en .'
    pattern = re.compile(pattern)
    with open(DBPedia_mapping_literals) as f:
        for line in tqdm(f):
            if pattern.match(line):
                article, p, label = pattern.findall(line)[0]
                article2names[article] |= {label}
    with open(DBPedia_infobox) as f:
        for line in tqdm(f):
            if pattern.match(line):
                article, p, label = pattern.findall(line)[0]
                article2names[article] |= {label}
    return article2names


CoNLL2003Categories = {"PER", "ORG", "LOC", "MISC"}


def terms_from_Wikipedia_for_cats(cats: List[str]) -> List[str]:
    # Get Wikipedia Articles corresponding DBPedia Concepts
    cat2articles = defaultdict(set)
    with open(DBPedia_instance_type) as f:
        for line in f:
            line = line.strip().split()
            cat2articles[line[2]] |= {line[0]}
            assert len(line) == 4
    articles = set()
    for c in cats:
        articles |= cat2articles[c]
    del cat2articles

    # Expand Wikipedia Articles using Redirect
    article2redirects = defaultdict(set)
    with open(DBPedia_redirect) as f:
        for line in tqdm(f):
            s, p, o, _ = line.strip().split()
            article2redirects[o] |= {s}

    # Get Names for Wikipedia Articles
    article2names = get_article2names()

    terms = []
    for article in tqdm(articles):
        expanded_articles = {article}
        expanded_articles |= article2redirects[article]
        for a in expanded_articles:
            terms += list(article2names[a])
    return terms


def get_names_from_entities(entities: Set[str]):
    logger.info("loading entity2names")
    # Translate DBPedia Entities into string by
    # DBPedia_WD_labels
    # DBPedia_WD_alias
    names = set()
    pattern = (
        "(<[^>]+>) "
        + "<http://www.w3.org/2000/01/rdf-schema#label>"
        + ' "([^"]+)"@en .'
    )
    pattern = re.compile(pattern)
    with open(DBPedia_WD_labels) as f:
        for line in tqdm(f, total=276814263):
            if pattern.match(line):
                entity, label = pattern.findall(line)[0]
                if entity in entities:
                    names.add(label)

    pattern = "(<[^>]+>) " + "<http://dbpedia.org/ontology/alias>" + ' "([^"]+)"@en .'
    pattern = re.compile(pattern)
    with open(DBPedia_WD_alias) as f:
        for line in tqdm(f, total=18641218):
            if pattern.match(line):
                entity, label = pattern.findall(line)[0]
                if entity in entities:
                    names.add(label)
    # DBPedia_WD_labels
    # DBPedia_WD_alias
    return names


def terms_from_Wikidata_for_cats(cats: List[str]) -> List[str]:
    # Get DBPedia Entities using
    # DBPedia_WD_instance_type
    # DBPedia_WD_SubClassOf
    entities = set()
    with open(DBPedia_WD_instance_type) as f:
        for line in f:
            ent, _, cl, _ = line.split()
            if cl in cats:
                entities.add(ent)
    with open(DBPedia_WD_SubClassOf) as f:
        for line in f:
            ent, _, cl, _ = line.split()
            if cl in cats:
                entities.add(ent)
    terms = get_names_from_entities(entities)
    return list(terms)


DBPediaCategories = load_DBPediaCategories()


def load_DBPedia_terms(names=["Agent"]) -> Set:
    assert all(name in DBPediaCategories for name in names)

    parent2children = load_dbpedia_parent2descendants()
    remained_category = {"<http://dbpedia.org/ontology/%s>" % name for name in names}
    descendants = set()
    while remained_category:
        parent = remained_category.pop()
        descendants.add(parent)
        children = parent2children[parent]
        remained_category |= children - descendants
    del parent2children

    terms = terms_from_Wikidata_for_cats(descendants)
    return set(terms)


CoNLL2003ToDBPediaCategoryMapper = {
    # NOTE: MISCはこれらいずれにも属さないカテゴリとする
    "PER": {"Person"},
    "ORG": {"Organization"},
    "LOC": {"Place"},
}


def load_CoNLL2003_terms(name="PER") -> Set:
    terms = set()
    if name == "MISC":
        mapped_categories = {
            correspond_cat
            for _, correspond_cats in CoNLL2003ToDBPediaCategoryMapper.items()
            for correspond_cat in correspond_cats
        }
        complement_categories = get_dbpedia_negative_cats_from_focus_cats(
            mapped_categories
        )
        terms = load_DBPedia_terms(complement_categories)
    else:
        terms = load_DBPedia_terms(CoNLL2003ToDBPediaCategoryMapper[name])
    return terms
