# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/

import os
import sys
from SPARQLWrapper import SPARQLWrapper, JSON
from logging import getLogger
from tqdm import tqdm
from p_tqdm import p_map
import json

logger = getLogger(__name__)

endpoint_url = "http://localhost:8890/sparql"
prefixes = """PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX dc: <http://purl.org/dc/terms/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>"""


def get_results(endpoint_url, query):
    user_agent = "WDQS-example Python/%s.%s" % (
        sys.version_info[0],
        sys.version_info[1],
    )
    # TODO adjust user agent; see https://w.wiki/CX6
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


def get_names_from_item_uri(
    uri: str = "http://dbpedia.org/resource/Major_League_Football",
):
    pass
    query = (
        prefixes
        + """
    SELECT DISTINCT ?label
    WHERE
    {
        {<%s> rdfs:label/<http://xmlns.com/foaf/0.1/name>/<http://dbpedia.org/property/name> ?label .} 
        UNION {
            ?redirectPage dbo:wikiPageRedirects* <%s> .
            ?redirectPage rdfs:label/<http://xmlns.com/foaf/0.1/name>/<http://dbpedia.org/property/name> ?label .
        }
    }""".replace(
            "%s", uri
        )
    )
    query = (
        prefixes
        + """
    SELECT DISTINCT ?label
    WHERE {
        {<%s> rdfs:label|<http://xmlns.com/foaf/0.1/name>|<http://dbpedia.org/property/name> ?label .} 
        UNION {
            ?redirectPage dbo:wikiPageRedirects <%s> .
            ?redirectPage rdfs:label|<http://xmlns.com/foaf/0.1/name>|<http://dbpedia.org/property/name> ?label .
        }
    }""".replace(
            "%s", uri
        )
    )
    terms = []
    results = get_results(endpoint_url, query)
    for binding in results["results"]["bindings"]:
        if "xml:lang" in binding["label"] and binding["label"]["xml:lang"] == "en":
            terms.append(binding["label"]["value"])
    return terms


def get_descendants(category: str = "dbo:Person"):
    query = (
        prefixes
        + """
    SELECT DISTINCT ?item
    WHERE
    {
        ?item rdf:type/rdfs:subClassOf* %s .
    }"""
    )
    results = get_results(endpoint_url, query % category)
    items = []
    for result in results["results"]["bindings"]:
        item = result["item"]
        if item["type"] == "uri":
            items.append(item["value"])
    termss = p_map(get_names_from_item_uri, items)
    # get_names_from_item_uri(items[1297])
    # terms = [get_names_from_item_uri(uri) for uri in tqdm(items)]
    return [term for terms in termss for term in terms]


def get_descendants_for_wiki_cat(
    category: str = "http://dbpedia.org/resource/Category:Products",
):
    categories_buffer = "data/buffer/%s.json" % category.split("/")[-1]
    if not os.path.exists(categories_buffer):
        skos_broader_query = (
            prefixes
            + """SELECT distinct * WHERE {
                ?item skos:broader <%s>
                }"""
        )
        categories = set()
        current_unchecked_categories = set([category])
        new_unchecked_categories = set()
        while current_unchecked_categories:
            logger.info(
                "# categories: %d, # unchecked_categories: %d"
                % (len(categories), len(current_unchecked_categories))
            )
            resultss = p_map(
                get_results,
                [endpoint_url] * len(current_unchecked_categories),
                [skos_broader_query % cat for cat in current_unchecked_categories],
                num_cpus=96,
            )
            results = [
                item for result in resultss for item in result["results"]["bindings"]
            ]
            # for cat in tqdm(current_unchecked_categories):
            #     results = get_results(endpoint_url, query % cat)
            for result in results:
                if result["item"]["type"] == "uri":
                    new_cat = result["item"]["value"]
                    if new_cat not in categories:
                        new_unchecked_categories.add(new_cat)
            categories |= set(current_unchecked_categories)
            current_unchecked_categories = new_unchecked_categories - categories
        with open(categories_buffer, "w") as f:
            json.dump(list(categories), f)
    with open(categories_buffer) as f:
        categories = set(json.load(f))
    dc_subject_query = (
        prefixes
        + """SELECT distinct ?item WHERE {
             ?item dc:subject <%s> .
             }"""
    )
    resultss = p_map(
        get_results,
        [endpoint_url] * len(categories),
        [dc_subject_query % cat for cat in categories],
        num_cpus=96,
    )
    results = [item for result in resultss for item in result["results"]["bindings"]]
    uris = [r["item"]["value"] for r in results]
    termss = p_map(get_names_from_item_uri, uris)
    return [term for terms in termss for term in terms]


def load_product():
    if not os.path.exists("data/buffer/products.json"):
        terms = get_descendants_for_wiki_cat(
            "http://dbpedia.org/resource/Category:Products"
        )
        with open("data/buffer/products.json", "w") as f:
            json.dump(terms, f)
    with open("data/buffer/products.json") as f:
        terms = json.load(f)
    logger.info("product are loaded")
    return terms


def load_twitter_main_dictionary():
    buffer_path = "data/buffer/twitter_main_dict.json"
    if not os.path.exists(buffer_path):
        product = set(load_product())
        company = set(get_descendants("dbo:Company"))
        sportsteam = set(get_descendants("dbo:SportsTeam"))
        person = set(get_descendants("dbo:Person"))
        musicartist = set(get_descendants("dbo:Musician"))
        person -= musicartist
        tvshow = set(get_descendants("dbo:TelevisionShow"))
        movies = set(get_descendants("dbo:Film"))
        geo_loc = set(get_descendants("dbo:Place"))
        facility = set(get_descendants("dbo:ArchitecturalStructure"))
        geo_loc -= facility
        # productだけデカすぎる (というかWikidata使ってるので汚い...)
        product -= tvshow
        product -= movies
        product -= facility
        product -= sportsteam
        product -= company
        product -= person
        product -= geo_loc
        cat2terms = {
            "company": company,
            "sportsteam": sportsteam,
            "person": person,
            "musicartist": musicartist,
            "tvshow": tvshow,
            "movies": movies,
            "geo_loc": geo_loc,
            "facility": facility,
            "product": product,
        }
        duplicated_terms = set()
        for c1, t1 in cat2terms.items():
            for c2, t2 in cat2terms.items():
                if c1 != c2:
                    duplicated_terms |= t1 & t2
        cat2terms = {cat: terms - duplicated_terms for cat, terms in cat2terms.items()}
        term2cat = {term: cat for cat, terms in cat2terms.items() for term in terms}
        with open(buffer_path, "w") as f:
            json.dump(term2cat, f, indent=4)
    with open(buffer_path) as f:
        term2cat = json.load(f)
    return term2cat


def load_twitter_sibling_dictionary(sibling_compression="none"):
    cat2terms = dict()
    # cat2terms["fake_cat_political_organization"] = get_descendants("Q7210356")  # comment out because of "time out"
    # sibling of sportsteam
    cat2terms["fake_cat_sports_league"] = set(get_descendants("dbo:SportsLeague"))
    cat2terms["fake_cat_broadcaster"] = set(get_descendants("dbo:Broadcaster"))
    cat2terms["fake_government_agency"] = set(get_descendants("dbo:GovernmentAgency"))
    cat2terms["fake_legislature"] = set(get_descendants("dbo:Legislature"))
    cat2terms["fake_military_unit"] = set(get_descendants("dbo:MilitaryUnit"))
    cat2terms["fake_political_party"] = set(get_descendants("dbo:PoliticalParty"))
    cat2terms["fake_parliament"] = set(get_descendants("dbo:Parliament"))
    cat2terms["fake_trade_union"] = set(get_descendants("dbo:TradeUnion"))
    cat2terms["fake_educational_institution"] = set(
        get_descendants("dbo:EducationalInstitution")
    )
    cat2terms["fake_employers_organisation"] = set(
        get_descendants("dbo:EmployersOrganisation")
    )
    cat2terms["fake_non_profit_organisation"] = set(
        get_descendants("dbo:Non-ProfitOrganisation")
    )
    cat2terms["fake_religious_organisation"] = set(
        get_descendants("dbo:ReligiousOrganisation")
    )
    duplicated_terms = set()
    for k1, v1 in cat2terms.items():
        for k2, v2 in cat2terms.items():
            if k1 != k2:
                duplicated_terms |= v1 & v2
    for k in cat2terms:
        cat2terms[k] = list(set(cat2terms[k]) - duplicated_terms)
    term2cat = {term: cat for cat, terms in cat2terms.items() for term in terms}
    return term2cat
