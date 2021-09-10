import json
from typing import Dict, List, Tuple, Set
import mysql.connector as mydb
from dataclasses import dataclass
from tqdm import tqdm
from time import time
import datetime
from more_itertools import chunked
from tqdm import tqdm
import requests, json
from urllib import parse
from collections import defaultdict


@dataclass
class Term2CatArgs:
    with_fake_label: bool = False
    with_fake_sibilings: bool = False


def run_umls_mysql(sql):

    # 接続する
    con = mydb.connect(
        user="myuser",
        password="p4ssw0rd",
        # host="localhost",
        db="umls2020AA",
        charset="utf8",
    )

    # カーソルを取得する
    cur = con.cursor()

    # クエリを実行する
    cur.execute(sql)

    # 実行結果をすべて取得する
    rows = cur.fetchall()

    cur.close()
    con.close()
    return rows


def get_children_of_chis(cuis: Set):
    children = run_umls_mysql(
        'select distinct cui2 from MRREL where cui1 in (%s) and rel="CHD";'
        % ", ".join(['"%s"' % cui for cui in cuis])
    )
    children = [c[0] for c in children]
    return children


def get_descendants_cuis(cui):
    searched_cuis = set()
    unsearched_cuis = {cui}
    start_time = time()
    num_cpus = 10
    while len(unsearched_cuis) > 0:
        print("number of searched_cuis: %d" % len(searched_cuis))
        print("number of unsearched_cuis: %d" % len(unsearched_cuis))
        print("%.2f [s]" % (time() - start_time,))
        print(datetime.datetime.now())
        one_step_children = list()
        for chunk in tqdm(list(chunked(unsearched_cuis, 1024))):
            one_step_children += get_children_of_chis(chunk)

        searched_cuis |= set(unsearched_cuis)
        unsearched_cuis = set(one_step_children) - searched_cuis
    return searched_cuis


def cui2terms(cui):
    terms = []
    terms = run_umls_mysql(
        'select distinct str from MRCONSO where cui="%s" and lat="ENG" and sab not in ("HGNC", "OMIM", "NCI", "SNOMEDCT_US", "PDQ", "CHV", "LNC");'.replace(
            "%s", str(cui)
        )
    )
    terms = [term[0] for term in terms]
    return list(set(terms))


def cuis2terms(cuis):
    terms = []
    terms = run_umls_mysql(
        'select distinct str from MRCONSO where cui in (%s) and lat="ENG" and sab not in ("HGNC", "OMIM", "NCI", "SNOMEDCT_US", "PDQ", "CHV", "LNC");'.replace(
            "%s", ", ".join(['"%s"' % cui for cui in cuis])
        )
    )
    terms = [term[0] for term in terms]
    return list(set(terms))


import os


def load_protein_descendants_cuis():
    # T116: Amino Acid, Peptide, or Protein
    protein_path = "data/genia_dict/protein_descendants.txt"
    if not os.path.exists(protein_path):
        proteins = get_descendants_cuis("C0033684")
        with open(protein_path, "w", encoding="utf-8") as f:
            f.write("\n".join(proteins))
    else:
        with open(protein_path, "r", encoding="utf-8") as f:
            proteins = f.read().split("\n")
    return proteins


def load_protein_domain_cuis():
    protein_domain_path = "data/genia_dict/protein_domains.txt"
    if not os.path.exists(protein_domain_path):
        protein_domains = get_descendants_cuis("C1514562")
        with open(protein_domain_path, "w", encoding="utf-8") as f:
            f.write("\n".join(protein_domains))
    else:
        with open(protein_domain_path, "r", encoding="utf-8") as f:
            protein_domains = f.read().split("\n")
    return protein_domains


def load_protein_cuis():
    # T116: Amino Acid, Peptide, or Protein
    protein_path = "data/genia_dict/proteins.txt"
    if not os.path.exists(protein_path):
        protein_descendants = load_protein_descendants_cuis()
        T116 = run_umls_mysql(
            'select distinct MRCONSO.cui from MRCONSO join MRSTY on MRCONSO.cui=MRSTY.cui where MRSTY.tui="T116" and MRCONSO.lat="ENG"'
        )
        T116 = [x[0] for x in T116]
        proteins = set(protein_descendants) & set(T116)
        proteins |= set(load_protein_domain_cuis())
        T087 = set(load_TUI_cuis("T087"))  # Amino Acid Sequence (for Domain or Motief)
        acceptable_cuis = set(T116) | T087
        with open(protein_path, "w", encoding="utf-8") as f:
            f.write("\n".join(proteins & acceptable_cuis))
    else:
        with open(protein_path, "r", encoding="utf-8") as f:
            proteins = f.read().split("\n")
    return proteins


def load_DNA_cuis():
    gene_path = "data/genia_dict/DNAs.txt"
    if not os.path.exists(gene_path):
        genes = get_descendants_cuis("C0012854")
        dna_sequences = get_descendants_cuis("C0162326")
        chromesomes = get_descendants_cuis("C0008633")
        histones = get_descendants_cuis("C0019652")
        # T059 = set(load_T059_cuis()) # Laboratory Procedure
        # T047 = set(load_TUI_cuis('T047')) # Disease or Syndrome
        # T201 = set(load_TUI_cuis('T201')) # Clinical Attribute
        # T019 = set(load_TUI_cuis('T019')) # Congenital Abnormality
        # removed_cuis = T059 | T047 | T201 | T019
        T028 = set(load_TUI_cuis("T028"))  # Gene or Genome
        T114 = set(load_TUI_cuis("T114"))  # "Nucleic Acid, Nucleoside, or Nucleotide"
        T086 = set(load_TUI_cuis("T086"))  # Nucleotide Sequence
        T026 = set(load_TUI_cuis("T026"))  # Cell Component
        acceptable_cuis = T028 | T114 | T086 | T026
        with open(gene_path, "w", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    (genes | dna_sequences | (chromesomes - histones)) & acceptable_cuis
                )
            )
    else:
        with open(gene_path, "r", encoding="utf-8") as f:
            genes = f.read().split("\n")
    return genes


def load_RNA_cuis():
    RNA_path = "data/genia_dict/RNAs.txt"
    if not os.path.exists(RNA_path):
        RNA = get_descendants_cuis("C0035668")
        T114 = set(load_TUI_cuis("T114"))
        T086 = set(load_TUI_cuis("T086"))
        acceptable_cuis = T114 | T086
        with open(RNA_path, "w", encoding="utf-8") as f:
            f.write("\n".join(RNA & acceptable_cuis))
    else:
        with open(RNA_path, "r", encoding="utf-8") as f:
            RNA = f.read().split("\n")
    return RNA


def load_T025_cuis():
    T025_path = "data/genia_dict/T025.txt"
    if not os.path.exists(T025_path):
        T025 = run_umls_mysql(
            'select distinct MRCONSO.cui from MRCONSO join MRSTY on MRCONSO.cui=MRSTY.cui where MRSTY.tui="T025" and MRCONSO.lat="ENG"'
        )
        T025 = [x[0] for x in T025]
        with open(T025_path, "w", encoding="utf-8") as f:
            f.write("\n".join(T025))
    else:
        with open(T025_path, "r", encoding="utf-8") as f:
            T025 = f.read().split("\n")
    return T025


def load_TUI_cuis(tui="T025"):
    tui_path = "data/genia_dict/%s.txt" % tui
    if not os.path.exists(tui_path):
        tui2stn = dict(run_umls_mysql("select distinct tui, stn from MRSTY;"))
        tree_number = tui2stn[tui]
        descendants = [
            tui for tui, stn in tui2stn.items() if stn.startswith(tree_number)
        ]
        cuis = []
        for des in descendants:
            _cuis = run_umls_mysql(
                'select distinct MRCONSO.cui from MRCONSO join MRSTY on MRCONSO.cui=MRSTY.cui where MRSTY.tui="%s" and MRCONSO.lat="ENG"'
                % des
            )
            cuis += [x[0] for x in _cuis]
        with open(tui_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cuis))
    else:
        with open(tui_path, "r", encoding="utf-8") as f:
            cuis = f.read().split("\n")
    return cuis


def load_T059_cuis():
    # T059: Laboratory Procedure
    T059_path = "data/genia_dict/T059.txt"
    if not os.path.exists(T059_path):
        T059 = run_umls_mysql(
            'select distinct MRCONSO.cui from MRCONSO join MRSTY on MRCONSO.cui=MRSTY.cui where MRSTY.tui="T059" and MRCONSO.lat="ENG"'
        )
        T059 = [x[0] for x in T059]
        with open(T059_path, "w", encoding="utf-8") as f:
            f.write("\n".join(T059))
    else:
        with open(T059_path, "r", encoding="utf-8") as f:
            T059 = f.read().split("\n")
    return T059


def load_cell_line():
    cell_line_path = "data/genia_dict/cell_line.txt"
    ## (5) Concept: [C0007600]  Cultured Cell Line の descendants: cell_line
    if not os.path.exists(cell_line_path):
        cell_line = get_descendants_cuis("C0007600")
        with open(cell_line_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cell_line))
    else:
        with open(cell_line_path, "r", encoding="utf-8") as f:
            cell_line = f.read().split("\n")
    return cell_line


def cui2tui(cui):
    return run_umls_mysql('select tui from MRSTY where cui="%s"' % cui)[0][0]


from pathlib import Path


def load_cat2cuis():
    data_path = Path("data/genia_dict/cat2cuis.json")
    if not data_path.exists():
        # protein, DNA and RNA
        ## (1) T116: Amino Acid, Peptide, or Protein を取得する: protein
        proteins = load_protein_cuis()
        ## (3) RNA: Concept: [C0035668]  RNA の descendants を取得する: RNA
        RNA = load_RNA_cuis()
        ## (2)から(3)に含まれているものを取り除く
        DNA = load_DNA_cuis()
        proteins = list((set(proteins) - set(DNA)) - set(RNA))
        DNA = list(set(DNA) - set(RNA))

        cell_line = load_cell_line()
        ## (4) T025: Cell のついているもの: cell_type
        T025 = load_T025_cuis()
        cell_type = list(set(T025) - set(cell_line))
        # (1-5)の重複しているものを取り除く
        # cui_sets = [set(DNA), set(RNA), set(proteins), set(cell_type), set(cell_line)]
        cat2cuis = {
            "DNA": DNA,
            "RNA": RNA,
            "protein": proteins,
            "cell_type": cell_type,
            "cell_line": cell_line,
        }
        cat2cuis = {k: list(set(v)) for k, v in cat2cuis.items()}
        with open(data_path, "w") as f:
            json.dump(cat2cuis, f)
    with open(data_path, "r") as f:
        cat2cuis = json.load(f)
    return cat2cuis


def load_raw_cat2terms():
    data_path = "data/genia_dict/raw_cat2terms.json"
    if not os.path.exists(data_path):

        # termss = []
        # for cui_set in cui_sets:
        #     cui_set = cui_set
        #     terms = cuis2terms(cui_set)
        #     terms = [term for term in terms if "&#x7C;" not in term and "," not in term]
        #     termss += [terms]
        # raw_cat2terms = {
        #     "DNA": termss[0],
        #     "RNA": termss[1],
        #     "protein": termss[2],
        #     "cell_type": termss[3],
        #     "cell_line": termss[4],
        # }
        cat2cuis = load_cat2cuis()
        raw_cat2terms = dict()
        for cat, cuis in cat2cuis.items():
            terms = cuis2terms(cuis)
            terms = [term for term in terms if "&#x7C;" not in term and "," not in term]
            raw_cat2terms[cat] = cuis
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(raw_cat2terms, f)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_cat2terms = json.load(f)
    return raw_cat2terms


def load_all_cuis():
    all_cuis_path = "data/genia_dict/all_cuis.txt"
    if not os.path.exists(all_cuis_path):
        all_cuis = run_umls_mysql(
            'select distinct MRCONSO.cui from MRCONSO where MRCONSO.lat="ENG"'
        )
        all_cuis = [x[0] for x in all_cuis]
        with open(all_cuis_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_cuis))
    else:
        with open(all_cuis_path, "r", encoding="utf-8") as f:
            all_cuis = f.read().split("\n")
    return all_cuis


def load_raw_cat2terms_wothers():
    data_path = "data/genia_dict/raw_cat2terms_wothers.json"
    if not os.path.exists(data_path):
        # (0) "DNA", "RNA", "protein", "cell_type", "cell_line" の cuiを取得する
        proteins = load_protein_cuis()
        ## (3) RNA: Concept: [C0035668]  RNA の descendants を取得する: RNA
        RNA = load_RNA_cuis()
        ## (2)から(3)に含まれているものを取り除く
        DNA = load_DNA_cuis()
        proteins = list((set(proteins) - set(DNA)) - set(RNA))

        cell_line = load_cell_line()
        ## (4) T025: Cell のついているもの: cell_type
        T025 = load_T025_cuis()
        cell_type = list(set(T025) - set(cell_line))
        # T059: Laboratory Procedureをロードする
        T059 = load_T059_cuis()
        # (1) それらのcuiを全てまとめる
        cui_sets = [set(DNA), set(RNA), set(proteins), set(cell_type), set(cell_line)]

        in_label_cuis = set()
        for cui_set in cui_sets:
            in_label_cuis |= cui_set
        # (2) mysqlから全てのcuiを取得する
        all_cuis = load_all_cuis()
        # (3) (2) から (1)を除外する
        all_cuis = set(all_cuis) - in_label_cuis
        # all_cuis -= unnecessary_cuis
        # (4) cuiを語句に変換する
        cui_sets += [all_cuis]

        duplicated_cuis = set()
        for i1, s1 in enumerate(cui_sets):
            for i2, s2 in enumerate(cui_sets):
                if i1 > i2:
                    duplicated_cuis.update(s1 & s2)

        termss = []
        for cui_set in cui_sets:
            cui_set = cui_set - duplicated_cuis
            termss += [cuis2terms(cui_set)]
        # (5) return
        T059_terms = set(term.lower() for term in cuis2terms(set(T059)))
        duplicated_terms_w_T059 = set()
        for terms in termss[:-1]:
            terms = set(term.lower() for term in terms)
            duplicated_terms_w_T059 |= terms & T059_terms

        raw_cat2terms_wothers = {
            "DNA": termss[0],
            "RNA": termss[1],
            "protein": termss[2],
            "cell_type": termss[3],
            "cell_line": termss[4],
            "Others": [
                term
                for term in termss[5]
                if term.lower() not in duplicated_terms_w_T059
            ],
        }
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(raw_cat2terms_wothers, f)
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_cat2terms_wothers = json.load(f)
    return raw_cat2terms_wothers


def load_fake_cat2terms(sibilling_compression):
    buffer_file = "data/genia_dict/fake_cat2terms_%s.json" % str(sibilling_compression)
    if not os.path.exists(buffer_file):
        cat2cuis = dict()
        # sibling of compound
        cat2cuis["atom"] = load_TUI_cuis("T196")
        # sibling of organic chemical
        cat2cuis["inorganic_chemical"] = load_TUI_cuis("T197")
        # sibling of Nucleic Acid
        # cat2cuis["peptide"] = get_descendants_cuis("C0030956") # protein を含むので除外
        # cat2cuis["aino_acid_monomer"] = get_descendants_cuis("C0002520") # proteins を含むので除外
        cat2cuis["carbohydrate"] = get_descendants_cuis("C0007004")
        cat2cuis["lipid"] = get_descendants_cuis("C0023779")
        # sibling of DNA, RNA (one of Nucleic Acid)
        # cat2cuis["nucleotide"] = get_descendants_cuis("C0028630") # Polynucleotideを含むので除外
        # cat2cuis["polynucleotide"] = get_descendants_cuis("")  # DNA や RNA と 重複しそうなので除外
        # sibling of cell_type
        cat2cuis["body_part"] = get_descendants_cuis("T023")
        cat2cuis["cell_component"] = load_TUI_cuis("T026")
        cat2cuis["organism"] = get_descendants_cuis("T194")

        cat2terms = dict()
        for k, v in cat2cuis.items():
            cat2terms["fake_cat_%s" % k] = cuis2terms(v)
        with open(buffer_file, "w") as f:
            json.dump(cat2terms, f, indent=4)
    else:
        with open(buffer_file) as f:
            cat2terms = json.load(f)
    return cat2terms


from hashlib import md5
from functools import reduce
from logging import getLogger

logger = getLogger(__name__)


def load_term2cat(
    with_sibilling: bool = False,
    sibilling_compression: str = "none",
    only_fake: bool = False,
):
    args = [with_sibilling, sibilling_compression, only_fake]
    term2cat_path = (
        "data/genia_dict/term2cat_%s.json" % md5(str(args).encode()).hexdigest()
    )
    logger.info("using term2cat at %s" % term2cat_path)

    if not os.path.exists(term2cat_path):
        if not only_fake:
            raw_cat2terms = load_raw_cat2terms()
            if with_sibilling:
                fake_cat2terms = load_fake_cat2terms(sibilling_compression)
                # raw_cat2termsに含まれていない語句だけを取得する
                # fake_categoryはラベルの名称を変更しておく
                # raw_cat2termsを追加する
                true_terms = reduce(
                    lambda x, y: x | y, map(set, raw_cat2terms.values()), set()
                )
                true_terms = {t.lower() for t in true_terms}
                for k, v in fake_cat2terms.items():
                    raw_cat2terms[k] = list(
                        {term for term in v if term.lower() not in true_terms}
                    )
        else:
            focused_cat2terms = load_raw_cat2terms()
            focused_terms = set(
                [term.lower() for terms in focused_cat2terms.values() for term in terms]
            )
            fake_cat2terms = load_fake_cat2terms(sibilling_compression)
            raw_cat2terms = dict()
            for k, v in fake_cat2terms.items():
                raw_cat2terms[k] = list(
                    {term for term in v if term.lower() not in focused_terms}
                )

        # 3文字未満の語句を除外
        raw_cat2terms = {
            k: set(term for term in v if len(term) >= 4)
            for k, v in raw_cat2terms.items()
        }

        duplicated_terms = set()
        for i1, v1 in enumerate(raw_cat2terms.values()):
            for i2, v2 in enumerate(raw_cat2terms.values()):
                if i1 > i2:
                    duplicated_terms.update(v1 & v2)
        non_duplicated_rawcat2terms = {
            k: v - duplicated_terms for k, v in raw_cat2terms.items()
        }
        term2cat = {
            term: cat
            for cat, terms in non_duplicated_rawcat2terms.items()
            for term in terms
        }
        with open(term2cat_path, "w", encoding="utf-8") as f:
            json.dump(term2cat, f)
    else:
        with open(term2cat_path, "r", encoding="utf-8") as f:
            term2cat = json.load(f)
    return term2cat


def load_ambiguous_term2cat() -> Dict:
    # cat2termsをロードする
    cat2terms: dict = load_raw_cat2terms()
    from collections import defaultdict

    term2cats = defaultdict(list)
    # todo: term2catsに変換する
    for cat, terms in cat2terms.items():
        for term in terms:
            term2cats[term].append(cat)

    ambiguous_terms = []
    for term, cats in term2cats.items():
        if len(cats) >= 2:
            ambiguous_terms.append(term)
    return {term: cats for term, cats in term2cats.items() if term in ambiguous_terms}


def load_termend2cat():
    term2cat_path = "data/genia_dict/termend2cat.json"
    if not os.path.exists(term2cat_path):
        raw_cat2terms = load_raw_cat2terms()

        # 3文字未満の語句を除外
        raw_cat2terms = {
            k: set(term for term in v if len(term) >= 4)
            for k, v in raw_cat2terms.items()
        }

        from collections import Counter

        endswith_cat2terms = dict()
        for cat in raw_cat2terms:
            c = Counter(
                [
                    " ".join(term.split()[-i:]).lower()
                    for term in raw_cat2terms[cat]
                    for i in range(4)
                ]
            )
            endswith_cat2terms[cat] = set([term for term, v in c.most_common()])

        # tfidf based screening
        from collections import defaultdict
        from math import log

        df = defaultdict(lambda: 0)
        for cat, terms in endswith_cat2terms.items():
            for term in terms:
                df[term] += 1
        idf = dict()
        doc_num = len(endswith_cat2terms)
        for term in df:
            idf[term] = log(doc_num / df[term])
        tf = defaultdict(dict)
        for cat in raw_cat2terms:
            c = Counter(
                [
                    " ".join(term.split()[-i:]).lower()
                    for term in raw_cat2terms[cat]
                    for i in range(4)
                ]
            )
            s = sum(c.values())
            for term, count in c.items():
                tf[cat][term] = count / s
        tfidf = defaultdict(dict)
        for cat in tf:
            for term, tf_val in tf[cat].items():
                tfidf[cat][term] = tf_val * idf[term]
        thr = 10 ** (-5)
        # thr = 0
        raw_cat2terms = {
            cat: set([term for term, val in tfidf[cat].items() if val > thr])
            for cat in tfidf
        }

        duplicated_terms = set()
        duplicated_terms_dict = defaultdict(set)
        for i1, (c1, v1) in enumerate(raw_cat2terms.items()):
            for i2, (c2, v2) in enumerate(raw_cat2terms.items()):
                if i1 > i2:
                    duplicated_terms.update(v1 & v2)
                    for term in v1 & v2:
                        duplicated_terms_dict[term].add(c1)
                        duplicated_terms_dict[term].add(c2)
        non_duplicated_rawcat2terms = {
            k: v - duplicated_terms for k, v in raw_cat2terms.items()
        }
        term2cat = {
            term: cat
            for cat, terms in non_duplicated_rawcat2terms.items()
            for term in terms
        }
        with open(term2cat_path, "w", encoding="utf-8") as f:
            json.dump(term2cat, f)
    else:
        with open(term2cat_path, "r", encoding="utf-8") as f:
            term2cat = json.load(f)
    return term2cat


def load_term2cat_wothers():
    term2cat_path = "data/genia_dict/term2cat_wothers.json"
    if not os.path.exists(term2cat_path):
        raw_cat2terms = load_raw_cat2terms_wothers()

        # 3文字未満の語句を除外
        raw_cat2terms = {
            k: set(term for term in v if len(term) >= 4)
            for k, v in raw_cat2terms.items()
        }
        # 一定の長さ以上のものを除外
        raw_cat2terms = {
            k: set(term for term in v if len(term.split()) <= 5)
            for k, v in raw_cat2terms.items()
        }
        import inflect

        p = inflect.engine()
        raw_cat2terms = {k: set(terms) for k, terms in raw_cat2terms.items()}
        already_include_terms = set()
        for k, terms in raw_cat2terms.items():
            already_include_terms |= set(terms)
        for k, terms in raw_cat2terms.items():
            plts = set()
            for term in tqdm(terms):
                try:
                    plt = p.plural(term)
                except IndexError:
                    pass
                else:
                    plts.add(plt)
            raw_cat2terms[k] |= plts - already_include_terms

        duplicated_terms = set()
        for i1, v1 in enumerate(raw_cat2terms.values()):
            for i2, v2 in enumerate(raw_cat2terms.values()):
                if i1 > i2:
                    duplicated_terms.update(v1 & v2)
        non_duplicated_rawcat2terms = {
            k: v - duplicated_terms for k, v in raw_cat2terms.items()
        }
        term2cat = {
            term: cat
            for cat, terms in non_duplicated_rawcat2terms.items()
            for term in terms
        }
        with open(term2cat_path, "w", encoding="utf-8") as f:
            json.dump(term2cat, f)
    else:
        with open(term2cat_path, "r", encoding="utf-8") as f:
            term2cat = json.load(f)
    return term2cat


def translate_cat2cuis_cat2wdis(cat2cuis) -> Dict:
    pass
    url = "https://query.wikidata.org/sparql?format=json&query=select%20?s%20?sLabel%20?o{%20?s%20wdt:P2892%20?o.%20SERVICE%20wikibase:label%20{%20bd:serviceParam%20wikibase:language%20%22[AUTO_LANGUAGE],en%22.%20}%20}"
    # https://w.wiki/3DmR と同じことをする
    ret = requests.get(url)
    ret = json.loads(ret.text)
    cui2wdis = defaultdict(list)
    for line in ret["results"]["bindings"]:
        cui = line["o"]["value"]
        cui2wdis[cui].append(parse.urlparse(line["s"]["value"]).path.split("/")[-1])
    pass
    cat2pageids = defaultdict(list)
    for cat, cuis in cat2cuis.items():
        for cui in tqdm(cuis):
            if cui in cui2wdis:
                # 後で、wikipedia pageidにmappingする
                cat2pageids[cat] += cui2wdis[cui]
            else:
                pass
                terms = cui2terms(cui)
                for term in terms:
                    pages = json.loads(
                        requests.get(
                            "https://en.wikipedia.org/w/api.php?action=query&titles=%s&&redirects&format=json"
                            % term
                        ).text
                    )["query"]["pages"]
                    if "-1" not in pages:
                        if len(pages) > 1:
                            raise NotImplementedError
                        elif len(pages) == 1:
                            cat2pageids[cat].append(list(pages.values())[0]["pageid"])

        assert len(cat2pageids[cat]) > 0


if __name__ == "__main__":
    cat2terms = load_term2cat()
