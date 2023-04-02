

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

