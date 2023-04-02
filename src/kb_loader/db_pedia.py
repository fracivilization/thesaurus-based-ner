import sqlite3
import os
import re
from tqdm import tqdm
from hydra.utils import to_absolute_path
from functools import lru_cache
from typing import Dict, Set
import json
from src.utils.utils import SQliteJsonDict
from src.dataset.utils import get_ascendant_dbpedia_thesaurus_node

DBPedia_dir = "data/DBPedia"
AnchorTextSourcePath = os.path.join(DBPedia_dir, "anchor-text_lang=en.ttl")
DBPedia_instance_type = os.path.join(DBPedia_dir, "instance-types_lang=en_specific.ttl")
DBPedia_redirect = os.path.join(DBPedia_dir, "redirects_lang=en.ttl")
AnchorTextDBPath = "data/sqlite_dbs/anchor_text_term2cats.db"


@lru_cache(maxsize=None)
def load_dbpedia_entity2cats() -> Dict:
    output_path = to_absolute_path("data/DBPedia/entity2cats.db")
    if not os.path.exists(output_path):
        entity2cats = dict()
        with open(to_absolute_path(DBPedia_instance_type)) as f:
            for line in tqdm(f, total=7636009):
                # 例: line = "<http://dbpedia.org/resource/'Ara'ir> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://dbpedia.org/ontology/Settlement> ."
                line = line.strip().split()
                assert len(line) == 4
                assert line[1] == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
                entity, _, cat, _ = line
                if entity in entity2cats:
                    cats = json.loads(entity2cats[entity])
                    cats.append(cat)
                    entity2cats[entity] = json.dumps(cats)
                else:
                    entity2cats[entity] = json.dumps([cat])
        # Expand Wikipedia Articles using Redirect
        unfound_redirects = []
        with open(to_absolute_path(DBPedia_redirect)) as f:
            for line in tqdm(f, total=10338969):
                line = line.strip().split()
                assert len(line) == 4
                assert line[1] == "<http://dbpedia.org/ontology/wikiPageRedirects>"
                entity, _, redirect, _ = line
                if redirect in entity2cats:
                    new_cats = json.loads(entity2cats[redirect])
                    if entity in entity2cats:
                        old_cats = json.loads(entity2cats[entity])
                        cats = list(set(old_cats + new_cats))
                        entity2cats[entity] = json.dumps(cats)
                    else:
                        entity2cats[entity] = json.dumps(new_cats)
                else:
                    unfound_redirects.append((entity, redirect))
        while unfound_redirects:
            print("unfound redirects num: ", len(unfound_redirects))
            remained_redirects = []
            for entity, redirect in unfound_redirects:
                if redirect in entity2cats:
                    new_cats = json.loads(entity2cats[redirect])
                    if entity in entity2cats:
                        old_cats = json.loads(entity2cats[entity])
                        cats = list(set(old_cats + new_cats))
                        entity2cats[entity] = json.dumps(cats)
                    else:
                        entity2cats[entity] = json.dumps(new_cats)
                else:
                    remained_redirects.append((entity, redirect))
            if len(unfound_redirects) == len(remained_redirects):
                break
            unfound_redirects = remained_redirects
            # TODO: WeightedDoubleArrayDictで読み込むようにする
        db_entity2cats = SQliteJsonDict(output_path, commit_when_set_item=False)
        for entity, cats in tqdm(entity2cats.items()):
            db_entity2cats[entity] = json.loads(cats)
        db_entity2cats.commit()
    entity2cats = SQliteJsonDict(output_path)
    return entity2cats


def expand_dbpedia_cats(cats: Set[str]) -> Set:
    # 1. シソーラスの構造に応じてラベル集合L={l_i}_iをパスに展開
    #    各ラベルまでのパス上にあるノードをすべて集める
    #    PATHS = {l \in PATH(l_i)}_{i in L}
    expanded_cats = set()
    for cat in cats:
        expanded_cats |= set(get_ascendant_dbpedia_thesaurus_node(cat))
    return expanded_cats


class AnchorTextTerm2CatsDB:
    def __init__(self) -> None:
        if not os.path.exists(AnchorTextDBPath):
            self.create_db()
        self.con = sqlite3.connect(AnchorTextDBPath)
        self.cur = self.con.cursor()

    def create_anchor_text_table(self):
        print("Creating anchor_text table...")
        con = sqlite3.connect(AnchorTextDBPath)
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = true")
        cur.execute("CREATE TABLE entities (entity text PRIMARY KEY)")
        cur.execute(
            "CREATE TABLE anchor_text (entity text, label text, FOREIGN KEY(entity) REFERENCES entities(entity))"
        )

        pattern = (
            "(<[^>]+>) "
            + "<http://dbpedia.org/ontology/wikiPageWikiLinkText> "
            + '"(.+)"@en .'
        )
        pattern = re.compile(pattern)

        source_length = 224826849
        for line in tqdm(open(AnchorTextSourcePath), total=source_length):
            match = pattern.match(line)
            if match:
                entity = match.group(1)
                label = match.group(2)
                cur.execute(
                    "INSERT OR IGNORE INTO entities (entity) VALUES (?)", (entity,)
                )
                cur.execute(
                    "INSERT INTO anchor_text (entity, label) VALUES (?, ?)",
                    (entity, label),
                )

        cur.execute("CREATE INDEX label_index ON anchor_text (label)")
        cur.execute("CREATE INDEX label_entity_index ON anchor_text (label, entity)")
        con.commit()
        cur.close()
        con.close()

    def entities_expanded_cats_table(self):
        print("Creating entities_expanded_cats table...")
        con = sqlite3.connect(AnchorTextDBPath)
        cur = con.cursor()
        cur.execute("PRAGMA foreign_keys = true")
        cur.execute("CREATE TABLE categories (cat text PRIMARY KEY)")
        cur.execute(
            "CREATE TABLE entities_expanded_cats (id INTEGER PRIMARY KEY AUTOINCREMENT, entity text, cat text, FOREIGN KEY(entity) REFERENCES entities(entity), FOREIGN KEY(cat) REFERENCES categories(cat))"
        )
        entity2cats = load_dbpedia_entity2cats()
        for entity, cats in tqdm(entity2cats.items(), total=len(entity2cats)):
            cats = expand_dbpedia_cats(cats)
            for cat in cats:
                cur.execute("INSERT OR IGNORE INTO categories (cat) VALUES (?)", (cat,))
                cur.execute(
                    "INSERT OR IGNORE INTO entities (entity) VALUES (?)", (entity,)
                )
                cur.execute(
                    "INSERT INTO entities_expanded_cats (entity, cat) VALUES (?, ?)",
                    (entity, cat),
                )
        cur.execute(
            "CREATE INDEX entity_cat_index ON entities_expanded_cats (entity, cat)"
        )
        cur.execute(
            "CREATE INDEX entity_index_on_entities_expanded_cats ON entities_expanded_cats (entity)"
        )
        cur.execute(
            "CREATE INDEX cat_index_on_entities_expanded_cats ON entities_expanded_cats (cat)"
        )
        con.commit()
        cur.close()
        con.close()

    def create_db(self):
        self.create_anchor_text_table()
        self.entities_expanded_cats_table()
        self.create_label_entity_rank_top_20_table()
        self.create_label_entity_entity_count_category_table()

    def entity_count_ranking_by_label(self, label, limit=20):
        self.cur.execute(
            "SELECT entity, COUNT(*) FROM anchor_text WHERE label=? GROUP BY entity ORDER BY COUNT(*) DESC LIMIT ?",
            (label, limit),
        )
        cats, weights = zip(*self.cur.fetchall())
        return cats, weights

    def terms(self, chunk_size=10000000):
        cur = self.con.cursor()
        offset = 0
        while True:
            cur.execute(
                "SELECT DISTINCT label FROM anchor_text LIMIT ? OFFSET ?",
                (chunk_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for row in rows:
                yield row[0]
            offset += chunk_size
        cur.close()

    def create_label_entity_rank_top_20_table(self):
        print("Creating label_entity_rank_top_20 table...")
        cur = self.con.cursor()
        cur.execute("PRAGMA foreign_keys = true")
        cur.execute(
            "CREATE TABLE label_entity_rank_top_20 (label text, entity text, count integer, FOREIGN KEY(entity) REFERENCES entities(entity), PRIMARY KEY(label, entity))"
        )
        for label in tqdm(self.terms(), total=23134550):
            entities, counts = self.entity_count_ranking_by_label(label, limit=20)
            for entity, count in zip(entities, counts):
                cur.execute(
                    "INSERT INTO label_entity_rank_top_20 (label, entity, count) VALUES (?, ?, ?)",
                    (label, entity, count),
                )
        cur.execute(
            "CREATE INDEX entity_index_on_label_entity_rank_top_20 ON label_entity_rank_top_20 (entity)"
        )
        cur.execute(
            "CREATE INDEX label_index_on_label_entity_rank_top_20 ON label_entity_rank_top_20 (label)"
        )
        cur.execute(
            "CREATE INDEX count_index_on_label_entity_rank_top_20 ON label_entity_rank_top_20 (count)"
        )
        self.con.commit()
        cur.close()

    def create_label_entity_entity_count_category_table(self):
        print("Creating label_entity_entity_count_category table...")
        cur = self.con.cursor()
        cur.execute("PRAGMA foreign_keys = true")
        cur.execute(
            "CREATE TABLE label_entity_entity_count_category (label text, entity text, entity_count integer, cat text, FOREIGN KEY(entity) REFERENCES entities(entity), FOREIGN KEY(cat) REFERENCES categories(cat), PRIMARY KEY(label, entity, cat))"
        )
        self.cur.execute(
            """SELECT COUNT(*) FROM label_entity_rank_top_20
            INNER JOIN entities_expanded_cats ON label_entity_rank_top_20.entity = entities_expanded_cats.entity"""
        )
        total_count = self.cur.fetchone()[0]
        chunk_size = 10000000
        offset = 0
        table_row_tqdm = tqdm(
            total=total_count, desc="label_entity_entity_count_category"
        )
        while True:
            cur.execute(
                """SELECT label_entity_rank_top_20.label, label_entity_rank_top_20.entity, label_entity_rank_top_20.count, entities_expanded_cats.cat
                FROM label_entity_rank_top_20
                INNER JOIN entities_expanded_cats ON label_entity_rank_top_20.entity = entities_expanded_cats.entity
                LIMIT ? OFFSET ?""",
                (chunk_size, offset),
            )
            rows = cur.fetchall()
            if not rows:
                break

            for row in rows:
                cur.execute(
                    "INSERT INTO label_entity_entity_count_category (label, entity, entity_count, cat) VALUES (?, ?, ?, ?)",
                    row,
                )
                table_row_tqdm.update()
            offset += chunk_size
        self.con.commit()
        cur.close()

    def term_cats_and_weights(self):
        cur = self.con.cursor()
        for term in tqdm(self.terms(), total=23134550):
            cur.execute(
                """SELECT cat, SUM(entity_count) AS total_entity_count
                FROM label_entity_entity_count_category
                WHERE label=? GROUP BY label, cat ORDER BY total_entity_count DESC""",
                (term,),
            )
            rows = cur.fetchall()
            if not rows:
                continue
            cats, weights = zip(*rows)
            yield term, cats, weights

    def __len__(self):
        cur = self.con.cursor()
        cur.execute("SELECT COUNT(*) FROM anchor_text")
        return cur.fetchone()[0]
