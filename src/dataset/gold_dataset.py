import re
from datasets.info import DatasetInfo
from src.ner_model.chunker.abstract_model import Chunker
from src.utils.params import get_ner_dataset_features
from datasets import DatasetDict, Dataset
from src.dataset.term2cat.terms import CoNLL2003Categories
from typing import List, Dict, Tuple
import spacy
from logging import getLogger
from tqdm import tqdm
from collections import Counter, defaultdict
import os
from seqeval.metrics.sequence_labeling import get_entities
import json
from src.dataset.utils import (
    tui2ST,
    get_tui2ascendants,
    CATEGORY_SEPARATOR,
    NEGATIVE_CATEGORY_TEMPLATE,
)
import datasets

logger = getLogger(__name__)
nlp = spacy.load("en_core_sci_sm")


def singularize_by_target_cats(
    multi_label_ner_dataset: Dataset, positive_cats: List[str], negative_cats: List[str]
) -> List[Dict]:
    """着目クラスに応じてマルチクラスデータセットをシングルクラスに変換する
    返り値はconll形式の辞書のリストで返す
    e.g.
    入力
    multi_label_ner_dataset
    {
        "tokens": ["Dave", "eat", "an", "apple", "."],
        "starts": [0, 3],
        "ends": [1, 4],
        "labels": [["PER", "lawyer], ["Fruit"]]

    }
    positive_cats: ["PER"]

    出力
    {
        "tokens": ["Dave", "eat", "an", "apple", "."],
        "tags": ["B-PER", "O", "O", O", "O"]
    }
    """
    ret_conll = []
    target_cats = set(positive_cats + negative_cats)
    label_names = multi_label_ner_dataset.features["labels"].feature.feature.names
    for snt in multi_label_ner_dataset:
        ner_tags = ["O"] * len(snt["tokens"])
        for span_labels, s, e in zip(snt["labels"], snt["starts"], snt["ends"]):
            span_labels = set([label_names[l] for l in span_labels])
            remained_label = span_labels & target_cats
            if len(remained_label) == 1:
                remained_label = remained_label.pop()
                if remained_label in negative_cats:
                    remained_label = NEGATIVE_CATEGORY_TEMPLATE % remained_label
                for i in range(s, e):
                    if i == s:
                        ner_tags[i] = "B-%s" % remained_label
                    else:
                        ner_tags[i] = "I-%s" % remained_label
        ret_conll.append({"tokens": snt["tokens"], "tags": ner_tags})
    return ret_conll


def snt_split_conll(
    tokens: List[str], tags: List[str], split_ids: List[int]
) -> List[Dict]:
    tokenss = []
    tagss = []
    for s, e in zip(split_ids, split_ids[1:]):
        if tags[s].startswith("I-"):
            tokenss[-1] += tokens[s:e]
            tagss[-1] += tags[s:e]
        else:
            tokenss.append(tokens[s:e])
            tagss.append(tags[s:e])

    return [{"tokens": tok, "tags": tag} for tok, tag in zip(tokenss, tagss)]


def tokenize_with_spans(text: str, spans: List[str], snt_split: bool = False):
    doc = nlp(text)
    cs2ts = {w.idx: w.i for w in doc}  # character_start2token_start
    ce2te = {w.idx + len(w): w.i + 1 for w in doc}  # character_start2token_end
    tokens = [w.text for w in doc]

    def split_token_by_spans(cs2ts, ce2te, tokens, spans):
        new_cs2ts, new_ce2te, new_tokens = cs2ts, ce2te, tokens

        for span in spans:
            if not span["start"] in new_cs2ts:
                cs, ts = sorted(
                    [(cs, ts) for cs, ts in new_cs2ts.items() if cs < span["start"]],
                    key=lambda x: x[0],
                )[-1]
                ce, te = sorted(
                    [(ce, te) for ce, te in new_ce2te.items() if span["start"] < ce],
                    key=lambda x: x[0],
                )[0]
                new_tokens = (
                    new_tokens[:ts]
                    + [
                        new_tokens[ts][: span["start"] - cs],
                        new_tokens[ts][span["start"] - cs :],
                    ]
                    + new_tokens[te:]
                )
                new_cs2ts[span["start"]] = ts + 1
                new_ce2te[span["start"]] = te

                for cs, ts in new_cs2ts.items():
                    if cs > span["start"]:
                        new_cs2ts[cs] = ts + 1
                for ce, te in new_ce2te.items():
                    if ce > span["start"]:
                        new_ce2te[ce] = te + 1
            if not span["end"] in new_ce2te:
                cs, ts = sorted(
                    [(cs, ts) for cs, ts in new_cs2ts.items() if cs < span["end"]],
                    key=lambda x: x[0],
                )[-1]
                ce, te = sorted(
                    [(ce, te) for ce, te in new_ce2te.items() if span["end"] < ce],
                    key=lambda x: x[0],
                )[0]
                new_tokens = (
                    new_tokens[:ts]
                    + [
                        new_tokens[ts][: span["end"] - cs],
                        new_tokens[ts][span["end"] - cs :],
                    ]
                    + new_tokens[te:]
                )
                new_cs2ts[span["end"]] = ts + 1
                new_ce2te[span["end"]] = te

                for cs, ts in new_cs2ts.items():
                    if cs > span["end"]:
                        new_cs2ts[cs] = ts + 1
                for ce, te in new_ce2te.items():
                    if ce > span["end"]:
                        new_ce2te[ce] = te + 1
        return new_cs2ts, new_ce2te, new_tokens

    cs2ts, ce2te, tokens = split_token_by_spans(cs2ts, ce2te, tokens, spans)

    tok_spans = []
    for span in spans:
        token_start = cs2ts[span["start"]]
        token_end = ce2te[span["end"]]
        term = "".join(tokens[token_start:token_end])
        assert term == span["name"].replace(" ", "")
        tok_spans.append(
            {
                "start": token_start,
                "end": token_end,
                "name": span["name"],
                "labels": span["labels"],
            }
        )

    tags = ["O"] * len(tokens)
    for span in tok_spans:
        for i in range(span["start"], span["end"]):
            if i == span["start"]:
                tags[i] = "B-%s" % span["labels"]
            else:
                tags[i] = "I-%s" % span["labels"]
    if snt_split:
        conlls = snt_split_conll(
            tokens, tags, split_ids=[snt.start for snt in doc.sents]
        )
    else:
        conlls = [{"tokens": tokens, "tags": tags}]
    return conlls


def translate_pubtator_into_conll(
    pubtator_doc: str,
) -> Tuple[int, Dict]:
    """from pubtator format, split sentence and"""
    doc = pubtator_doc.split("\n")
    title, abstract = doc[0], doc[1]
    pmid = title[:8]
    title = title[11:]
    abstract = abstract[11:]
    spans = []
    title_spans = []
    abst_spans = []
    for span in doc[2:]:
        pmid, start, end, name, labels, cui = span.split("\t")
        start = int(start)
        end = int(end)
        spans.append(set(range(start, end)))
        # TODO: MultiLabelで過剰にスパンを除去してしまっているのでこの処理を除く、
        # ただしSingle Labelでラベルが重複しないようにする後処理をどこかで追加する
        if end <= len(title):
            title_spans.append(
                {"start": start, "end": end, "labels": labels, "name": name}
            )
        else:
            abst_spans.append(
                {
                    "start": start - len(title) - 1,
                    "end": end - len(title) - 1,
                    "labels": labels,
                    "name": name,
                }
            )
    # Check no span duplication,
    for i1, s1 in enumerate(spans):
        for i2, s2 in enumerate(spans):
            if i2 > i1:
                assert s1 & s2 == set()
    conlls = []
    conlls += tokenize_with_spans(title, title_spans)
    conlls += tokenize_with_spans(abstract, abst_spans, snt_split=True)
    return int(pmid), conlls


def translate_conll_into_dataset(
    conll_snts: Dict, ner_tag_names: List[str], desc: Dict = dict()
) -> Dataset:
    desc = json.dumps(desc)
    ret_dataset = Dataset.from_dict(
        {
            "tokens": [snt["tokens"] for snt in conll_snts],
            "ner_tags": [snt["tags"] for snt in conll_snts],
        },
        info=DatasetInfo(
            description=desc, features=get_ner_dataset_features(ner_tag_names)
        ),
    )
    return ret_dataset


def remove_span_duplication(docs: List[str]):
    screened_docs = []
    for doc in tqdm(docs):
        if doc != "":
            doc = doc.split("\n")
            title = doc[0]
            abstract = doc[1]
            spans = doc[2:]
            remained_spans = spans
            span_areas = []
            for span in spans:
                pmid, start, end, name, label, cui = span.split("\t")
                start, end = int(start), int(end)
                span_areas.append(set(range(start, end)))
            remove_spans = []
            for i1, s1 in enumerate(span_areas):
                for i2, s2 in enumerate(span_areas):
                    if i2 > i1:
                        if s1 & s2:
                            if s1 <= s2:
                                remove_spans.append(spans[i1])
                            elif s2 <= s1:
                                remove_spans.append(spans[i2])
                            else:
                                remove_spans.append(spans[i1])
                                remove_spans.append(spans[i2])
            for rs in remove_spans:
                if rs in remained_spans:
                    remained_spans.remove(rs)
            screened_docs.append("\n".join([title, abstract] + remained_spans))
    return screened_docs
    pass


def load_gold_datasets(
    positive_cats: str, negative_cats: str, input_dir: str, train_snt_num: int, random_seed: int,
) -> DatasetDict:
    # load remained cats
    positive_cats = positive_cats.split(CATEGORY_SEPARATOR)
    negative_cats = negative_cats.split(CATEGORY_SEPARATOR)

    # load dataset
    multi_label_ner = DatasetDict.load_from_disk(input_dir)

    singularized_ner = {
        key: singularize_by_target_cats(split, positive_cats, negative_cats)
        for key, split in multi_label_ner.items()
    }
    c = Counter(
        [
            tag
            for split in singularized_ner.values()
            for snt in split
            for tag in snt["tags"]
        ]
    )
    ner_tag_names = [tag for tag, _ in c.most_common()]
    dataset_dict = dict()
    for key, split in singularized_ner.items():
        if key == 'train':
            import random
            random.seed(random_seed)
            split = random.shuffle(split)[:train_snt_num]
        dataset_dict[key] = translate_conll_into_dataset(
            split,
            ner_tag_names,
            desc={
                "desc": "MedMentions Dataset focus on %s" % str(positive_cats),
                "positive_cats": positive_cats,
                "split": key,
            },
        )
    return DatasetDict(dataset_dict)


def translate_MedMentions_conll_into_msmlc_dataset(
    conll_snt: List[Dict],
    label_names: List[str],
    desc: Dict = dict(),
) -> Dataset:
    # TODO: multi span multi class datasetに変換する
    desc = json.dumps(desc)
    ret_dataset = defaultdict(list)
    tui2ascendants = get_tui2ascendants()
    for snt in conll_snt:
        ret_dataset["tokens"].append(snt["tokens"])
        starts, ends, labels = [], [], []
        for ls, s, e in get_entities(snt["tags"]):
            if ls == "UnknownType":
                continue
            starts.append(s), ends.append(e + 1)
            ascendant_labels = []
            for l in ls.split(","):
                ascendant_labels += tui2ascendants[l]
            labels.append(sorted(ascendant_labels))
        ret_dataset["starts"].append(starts)
        ret_dataset["ends"].append(ends)
        ret_dataset["labels"].append(labels)
        pass

    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "starts": datasets.Sequence(datasets.Value("int32")),
            "ends": datasets.Sequence(datasets.Value("int32")),
            "labels": datasets.Sequence(
                datasets.Sequence(datasets.ClassLabel(names=label_names))
            ),
        }
    )

    ret_dataset = Dataset.from_dict(
        ret_dataset,
        info=DatasetInfo(description=desc, features=features),
    )
    return ret_dataset


def load_MedMentions_gold_multi_label_ner_datasets(input_dir: str):
    # load dataset
    pubtator = os.path.join(input_dir, "corpus_pubtator.txt")
    with open(pubtator) as f:
        all_dataset = f.read().split("\n\n")

    screened_docs = remove_span_duplication(all_dataset)
    pmid2conll = dict()
    for doc in tqdm(screened_docs):
        pmid, conll = translate_pubtator_into_conll(doc)
        pmid2conll[pmid] = conll

    # Split Dataset
    train_pmids = os.path.join(input_dir, "corpus_pubtator_pmids_trng.txt")
    dev_pmids = os.path.join(input_dir, "corpus_pubtator_pmids_dev.txt")
    test_pmids = os.path.join(input_dir, "corpus_pubtator_pmids_test.txt")
    with open(train_pmids) as f:
        train_pmids = [int(line.strip()) for line in f]
    with open(dev_pmids) as f:
        dev_pmids = [int(line.strip()) for line in f]
    with open(test_pmids) as f:
        test_pmids = [int(line.strip()) for line in f]
    train_conll = [snt for pmid in train_pmids for snt in pmid2conll[pmid]]
    # train_conll = train_conll[:train_snt_num]
    dev_conll = [snt for pmid in dev_pmids for snt in pmid2conll[pmid]]
    test_conll = [snt for pmid in test_pmids for snt in pmid2conll[pmid]]
    label_names = sorted(tui2ST.keys())
    # if with_o:
    #     label_names = ["nc-O"] + label_names
    desc = {"desc": "MedMentions MSMLC Dataset"}
    dataset_dict = dict()
    desc["split"] = "train"
    dataset_dict["train"] = translate_MedMentions_conll_into_msmlc_dataset(
        train_conll, label_names, desc
    )
    desc["split"] = "validation"
    dataset_dict["validation"] = translate_MedMentions_conll_into_msmlc_dataset(
        dev_conll, label_names, desc
    )
    desc["split"] = "test"
    dataset_dict["test"] = translate_MedMentions_conll_into_msmlc_dataset(
        test_conll, label_names, desc
    )
    # describe positive_cat into datasetdict or dataset (describing into dataset dict is better)
    return DatasetDict(dataset_dict)


def translate_CoNLL2003_conll_into_msmlc_dataset(
    conll_snt: List[Dict],
    label_names: List[str],
    desc: Dict = dict(),
) -> Dataset:
    desc = json.dumps(desc)
    ret_dataset = defaultdict(list)
    for snt in conll_snt:
        ret_dataset["tokens"].append(snt["tokens"])
        starts, ends, labels = [], [], []
        for ls, s, e in get_entities(snt["tags"]):
            if ls == "UnknownType":
                continue
            starts.append(s), ends.append(e + 1)
            labels.append(ls.split(","))
        ret_dataset["starts"].append(starts)
        ret_dataset["ends"].append(ends)
        ret_dataset["labels"].append(labels)

    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "starts": datasets.Sequence(datasets.Value("int32")),
            "ends": datasets.Sequence(datasets.Value("int32")),
            "labels": datasets.Sequence(
                datasets.Sequence(datasets.ClassLabel(names=label_names))
            ),
        }
    )

    ret_dataset = Dataset.from_dict(
        ret_dataset,
        info=DatasetInfo(description=desc, features=features),
    )
    return ret_dataset


def load_CoNLL2003_gold_multi_label_ner_datasets(input_dir):
    # load dataset
    # TODO: まずCoNLL2003データを読み込む
    # TODO: CoNLL2003データをtranslate_conll_into_msmlc_datasetで変換する
    split_key2file = {
        "train": "train.txt",
        "validation": "valid.txt",
        "test": "test.txt",
    }
    key2conll = dict()
    for key, file in split_key2file.items():
        with open(os.path.join(input_dir, file)) as f:
            all_dataset = f.read().split("\n\n")
            data_split = []
            for snt in all_dataset[1:]:  # NOTE 最初の行の-DOCSTART-を取り除く
                conll_snt = {"tokens": [], "tags": []}
                for word in snt.split("\n"):
                    if word:
                        token, _, _, tag = word.split(" ")
                    conll_snt["tokens"].append(token)
                    conll_snt["tags"].append(tag)
                data_split.append(conll_snt)
        key2conll[key] = data_split

    label_names = list(CoNLL2003Categories)
    label_names.sort()
    desc = {"desc": "CoNLL2003 MSMLC Dataset"}
    dataset_dict = dict()
    for key, conll in key2conll.items():
        desc["split"] = key
        dataset_dict[key] = translate_CoNLL2003_conll_into_msmlc_dataset(
            conll, label_names, desc
        )

    return DatasetDict(dataset_dict)
