import os
from typing import Dict, Set
from tqdm import tqdm
from collections import defaultdict
from rdflib import Graph, URIRef
import re

umls_dir = "data/2021AA-full/data/2021AA"
mrsty = os.path.join(umls_dir, "META", "MRSTY.RRF")
mrconso = os.path.join(umls_dir, "META", "MRCONSO.RRF")
srdef = os.path.join(umls_dir, "NET", "SRDEF")
DBPedia_dir = "data/DBPedia"
DBPedia_ontology = os.path.join(DBPedia_dir, "ontology--DEV_type=parsed_sorted.nt")
DBPedia_instance_type = os.path.join(DBPedia_dir, "instance-types_lang=en_specific.ttl")
DBPedia_mapping_literals = os.path.join(
    DBPedia_dir, "mappingbased-literals_lang=en.ttl"
)
DBPedia_infobox = os.path.join(DBPedia_dir, "infobox-properties_lang=en.ttl")
DBPedia_redirect = os.path.join(DBPedia_dir, "redirects_lang=en.ttl")


def load_TUI_terms(tui="T025"):
    tui2stn: Dict[str, str] = dict()
    with open(srdef) as f:
        for line in f:
            line = line.strip().split("|")
            if line[1] not in tui2stn:
                tui2stn[line[1]] = line[3]
            else:
                raise NotImplementedError
    root_stn = tui2stn[tui]
    include_tuis = {tui for tui, stn in tui2stn.items() if stn.startswith(root_stn)}
    with open(mrsty) as f:
        cuis = [
            line.strip().split("|")[0]
            for line in f
            if line.split("|")[1] in include_tuis
        ]
    cuis = set(cuis)
    cui2terms = defaultdict(set)
    with open(mrconso) as f:
        for line in tqdm(f, total=16132274):
            if line.split("|")[1] == "ENG":
                cui2terms[line.split("|")[0]] |= {line.strip().split("|")[14]}
    terms = {term for cui in tqdm(cuis) for term in cui2terms[cui]}
    return terms


def get_article2names():
    import pickle

    output = os.path.join(DBPedia_dir, "article2names.pkl")
    if not os.path.exists(output):
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
        with open(output, "wb") as f:
            pickle.dump(article2names, f)
    with open(output, "rb") as f:
        article2names = pickle.load(f)
    return article2names


DBPedia_categories = {"Agent"}


def load_DBPedia_terms(name="Agent") -> Set:
    assert name in DBPedia_categories

    # Get DBPedia Concepts
    g = Graph()
    g.parse(DBPedia_ontology)
    s, p, o = next(g.__iter__())
    parent2children = defaultdict(set)
    for s, p, o in g:
        if isinstance(o, URIRef) and p.n3() in {
            "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>",
            "<http://www.w3.org/2000/01/rdf-schema#subClassOf>",
        }:
            parent2children[o.n3()] |= {s.n3()}
    remained_category = {"<http://dbpedia.org/ontology/%s>" % name}
    descendants = set()
    while remained_category:
        parent = remained_category.pop()
        descendants.add(parent)
        children = parent2children[parent]
        remained_category |= children - descendants
    del parent2children

    # Get Wikipedia Articles corresponding DBPedia Concepts
    cat2articles = defaultdict(set)
    with open(DBPedia_instance_type) as f:
        for line in f:
            line = line.strip().split()
            cat2articles[line[2]] |= {line[0]}
            assert len(line) == 4
    articles = set()
    for d in descendants:
        articles |= cat2articles[d]
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
    return set(terms)
