import re
from datasets.info import DatasetInfo
from src.ner_model.chunker.abstract_model import Chunker
from src.utils.params import get_ner_dataset_features
from datasets import DatasetDict, Dataset
from src.dataset.term2cat.terms import get_descendants_TUIs
from typing import List, Dict, Tuple
import spacy
from logging import getLogger
from tqdm import tqdm
from collections import Counter, defaultdict
import os
from seqeval.metrics.sequence_labeling import get_entities
import json
from src.dataset.utils import tui2ST, get_tui2ascendants
import datasets

logger = getLogger(__name__)
nlp = spacy.load("en_core_sci_sm")


def remain_only_focus_cats(
    pubtator_datasets: List[str], focus_cats: List[str]
) -> List[str]:
    dec2root = {dec: tui for tui in focus_cats for dec in get_descendants_TUIs(tui)}
    remained_cats = set(list(dec2root.keys()))

    # remain only remained cats
    screened_docs = []
    for doc in pubtator_datasets:
        doc = doc.split("\n")
        title = doc[0]
        abstract = doc[1]
        spans = doc[2:]
        remained_spans = []
        for span in spans:
            pmid, start, end, name, labels, cui = span.split("\t")
            labels = set(labels.split(","))
            remained_labels = labels & remained_cats
            if remained_labels:
                remained_spans.append(
                    "%s\t%s\t%s\t%s\t%s\t%s"
                    % (
                        pmid,
                        start,
                        end,
                        name,
                        ",".join(set(dec2root[l] for l in remained_labels)),
                        cui,
                    )
                )
        screened_docs.append("\n".join([title, abstract] + remained_spans))
    return screened_docs


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
        try:
            assert len(labels.split(",")) == 1
        except AssertionError:
            logger.info("span: %s is removed because the label is duplicated." % name)
            continue
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
    conll_snt: Dict, ner_tag_names: List[str], desc: Dict = dict()
) -> Dataset:
    desc = json.dumps(desc)
    ret_dataset = Dataset.from_dict(
        {
            "tokens": [snt["tokens"] for snt in conll_snt],
            "ner_tags": [snt["tags"] for snt in conll_snt],
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
    focus_cats: str, input_dir: str, train_snt_num: int
) -> DatasetDict:
    # load remained cats
    focus_cats = focus_cats.split("_")

    # load dataset
    pubtator = os.path.join(input_dir, "corpus_pubtator.txt")
    with open(pubtator) as f:
        all_dataset = f.read().split("\n\n")

    screened_docs = remain_only_focus_cats(
        all_dataset[:-1], focus_cats
    )  # remove no nontent dataset by [:-1]
    screened_docs = remove_span_duplication(screened_docs)
    pmid2conll = dict()
    for doc in screened_docs:
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
    train_conll = train_conll[:train_snt_num]
    dev_conll = [snt for pmid in dev_pmids for snt in pmid2conll[pmid]]
    test_conll = [snt for pmid in test_pmids for snt in pmid2conll[pmid]]
    c = Counter(
        [tag for doc in pmid2conll.values() for snt in doc for tag in snt["tags"]]
    )
    ner_tag_names = [tag for tag, _ in c.most_common()]
    desc = {
        "desc": "MedMentions Dataset focus on %s" % str(focus_cats),
        "focus_cats": focus_cats,
    }
    dataset_dict = dict()
    desc["split"] = "train"
    dataset_dict["train"] = translate_conll_into_dataset(
        train_conll, ner_tag_names, desc
    )
    desc["split"] = "validation"
    dataset_dict["validation"] = translate_conll_into_dataset(
        dev_conll, ner_tag_names, desc
    )
    desc["split"] = "test"
    dataset_dict["test"] = translate_conll_into_dataset(test_conll, ner_tag_names, desc)
    # describe focus_cat into datasetdict or dataset (describing into dataset dict is better)
    return DatasetDict(dataset_dict)


def translate_conll_into_msmlc_dataset(
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
        for l, s, e in get_entities(snt["tags"]):
            if l == "UnknownType":
                continue
            starts.append(s), ends.append(e + 1)
            ascendant_labels = tui2ascendants[l]
            labels.append(ascendant_labels)
        labeled_spans = set(zip(starts, ends))

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


def load_gold_multi_label_ner_datasets(input_dir: str):
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
    desc = {"desc": "MSMLC Dataset"}
    dataset_dict = dict()
    desc["split"] = "train"
    dataset_dict["train"] = translate_conll_into_msmlc_dataset(
        train_conll, label_names, desc
    )
    desc["split"] = "validation"
    dataset_dict["validation"] = translate_conll_into_msmlc_dataset(
        dev_conll, label_names, desc
    )
    desc["split"] = "test"
    dataset_dict["test"] = translate_conll_into_msmlc_dataset(
        test_conll, label_names, desc
    )
    # describe focus_cat into datasetdict or dataset (describing into dataset dict is better)
    return DatasetDict(dataset_dict)
