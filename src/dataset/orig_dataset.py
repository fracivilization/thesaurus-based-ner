from pathlib import Path

import datasets
from src.utils.params import get_ner_dataset_features, task_name2ner_label_names
from typing import Dict, List
import colt
from datasets import DatasetDict
import subprocess
import os
from loguru import logger
from transformers import AutoTokenizer
from datasets import Dataset
import pickle
from collections import defaultdict


def download_jnlpba_dataset():
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/jnlpba"):
        os.mkdir("data/jnlpba")
    if not os.path.exists("data/jnlpba/Genia4ERtraining"):
        pass
        subprocess.run(
            [
                "wget",
                "-P",
                "data/jnlpba",
                "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Train/Genia4ERtraining.tar.gz",
            ]
        )
        os.mkdir("data/jnlpba/Genia4ERtraining")
        subprocess.run(
            [
                "tar",
                "xzvf",
                "data/jnlpba/Genia4ERtraining.tar.gz",
                "-C",
                "data/jnlpba/Genia4ERtraining",
            ]
        )
        os.remove("data/jnlpba/Genia4ERtraining.tar.gz")
    if not os.path.exists("data/jnlpba/Genia4ERtest"):
        pass
        subprocess.run(
            [
                "wget",
                "-P",
                "data/jnlpba",
                "http://www.nactem.ac.uk/GENIA/current/Shared-tasks/JNLPBA/Evaluation/Genia4ERtest.tar.gz",
            ]
        )
        os.mkdir("data/jnlpba/Genia4ERtest")

        subprocess.run(
            [
                "tar",
                "xzvf",
                "data/jnlpba/Genia4ERtest.tar.gz",
                "-C",
                "data/jnlpba/Genia4ERtest",
            ]
        )
        os.remove("data/jnlpba/Genia4ERtest.tar.gz")


def init_genia_tagger():
    if not os.path.exists("lib"):
        pass
    if not os.path.exists("lib/geniatagger-3.0.2"):
        subprocess.run(
            [
                "wget",
                "-P",
                "lib",
                "http://www.nactem.ac.uk/GENIA/tagger/geniatagger-3.0.2.tar.gz",
            ]
        )
        subprocess.run(["tar", "xzvf", "lib/geniatagger-3.0.2.tar.gz", "-C", "lib"])
        os.chdir("lib/geniatagger-3.0.2")
        subprocess.run("make")
        os.chdir("../..")
        os.remove("lib/geniatagger-3.0.2.tar.gz")


def remove_space_token(tokens, ner_tags):
    new_tokens = []
    new_ner_tags = []
    for token, ner_tag in zip(tokens, ner_tags):
        new_token = []
        new_ner_tag = []
        for tok, nt in zip(token, ner_tag):
            if tok != " ":
                new_token.append(tok)
                new_ner_tag.append(nt)
            else:
                logger.info('There are space in snt: "%s".' % " ".join(token))
        new_tokens.append(new_token)
        new_ner_tags.append(new_ner_tag)
    return new_tokens, new_ner_tags


def load_document_wise_jnlpba_dataset() -> Dict[str, List]:
    import pickle

    jnlpba_dir = "data/jnlpba"
    output_path = Path(jnlpba_dir).joinpath("doc_wise_dataset.pkl")
    if not output_path.exists():
        files = {
            "train": "data/jnlpba/Genia4ERtraining/Genia4ERtask2.iob2",
            "validation": "data/jnlpba/Genia4ERtraining/sampletest2.iob2",
            "test": "data/jnlpba/Genia4ERtest/Genia4EReval2.iob2",
        }
        parsed_data = {}
        for key, file in files.items():
            docs = []
            doc = []
            snt = []
            with open(file) as f:
                for line in f:
                    if line.startswith("###MEDLINE:"):
                        if doc:
                            docs.append(doc)
                        doc = []
                    elif line == "\n":
                        if snt:
                            words, tags = zip(*snt)
                            doc.append(snt)
                            snt = []
                    else:
                        snt.append(line.strip().split("\t"))
                else:
                    pass
            parsed_data[key] = docs
        for key, docs in parsed_data.items():
            snts = [list(zip(*snt))[0] for doc in docs for snt in doc]
            with open("lib/geniatagger-3.0.2/tmp.txt", "w") as f:
                f.write("\n".join([" ".join(snt) for snt in snts]))
            os.chdir("lib/geniatagger-3.0.2/")
            os.system("./geniatagger tmp.txt > out.txt")
            os.chdir("../..")
            with open("lib/geniatagger-3.0.2/out.txt") as f:
                tagged_snts = f.read().split("\n\n")[:-1]
            assert len(tagged_snts) == len(snts)
            poss = [[w.split("\t")[2] for w in snt.split("\n")] for snt in tagged_snts]
            new_poss = []
            for sntid, (snt, pos) in enumerate(zip(snts, poss)):
                if len(snt) != len(pos):
                    logger.info("Allert! there is a POS difference")
                new_poss.append(pos[: len(snt)])
            parsed_data[key] = {"tokens": [], "ner_tags": [], "poss": []}
            snt_id = 0
            for doc in docs:
                tokens = []
                ner_tags = []
                poss = []
                for snt in doc:
                    words, tags = zip(*snt)
                    tokens.append(words)
                    ner_tags.append(tags)
                    poss.append(new_poss[snt_id])
                    snt_id += 1
                parsed_data[key]["tokens"].append(tokens)
                parsed_data[key]["ner_tags"].append(ner_tags)
                parsed_data[key]["poss"].append(poss)
        output_dict = {
            "train": parsed_data["train"],
            "validation": parsed_data["validation"],
            "test": parsed_data["test"],
        }
        with open(output_path, "wb") as f:
            pickle.dump(output_dict, f)
    with open(output_path, "rb") as f:
        output_dict = pickle.load(f)

    return output_dict


def download_twitter_dataset():
    if not os.path.exists("data/twitter_distant"):
        subprocess.run(
            [
                "svn",
                "export",
                "https://github.com/cliang1453/BOND/trunk/dataset/twitter_distant",
                "data/twitter_distant",
            ]
        )
        pass


def load_twitter_dataset():
    import json

    twitter_dir = Path("data/twitter_distant")
    with open(twitter_dir.joinpath("tag_to_id.json")) as f:
        tag2id = json.load(f)
    id2tag = {v: k for k, v in tag2id.items()}
    key2file = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
    parsed_data = dict()
    for key, file in key2file.items():
        parsed_data[key] = {"tokens": [], "ner_tags": [], "poss": []}
        with open(twitter_dir.joinpath(file)) as f:
            file = json.load(f)
        for snt in file:
            parsed_data[key]["tokens"].append([snt["str_words"]])
            ner_tags = [id2tag[tag] for tag in snt["tags"]]
            parsed_data[key]["ner_tags"].append([ner_tags])
    return parsed_data


from spacy.tokens import Doc


def custom_tokenizer(text):
    tokens = []

    # your existing code to fill the list with tokens

    # replace this line:
    return tokens

    # with this:
    return Doc(nlp.vocab, tokens)


import spacy


def add_pos(orig_dataset: Dict):
    pass
    nlp = spacy.load("en_core_web_sm")
    tokens_dict = dict()

    def custom_tokenizer(text):
        if text in tokens_dict:
            return Doc(nlp.vocab, tokens_dict[text])
        else:
            raise ValueError("No tokenization available for input.")

    nlp.tokenizer = custom_tokenizer
    for key, split in orig_dataset.items():
        for doc_id, doc in enumerate(split["tokens"]):
            poss = []
            for snt in doc:
                text = " ".join(snt)
                tokens_dict[text] = snt
                spacy_doc = nlp(text)
                assert len(spacy_doc) == len(snt)
                poss += [w.tag_ for w in spacy_doc]
            split["poss"].append(poss)
    return orig_dataset


def snt_tokenize_ner_datasets(ner_datasets: DatasetDict):
    pass
    ner_features = get_ner_dataset_features(
        ner_datasets["test"].features["ner_tags"].feature.names
    )
    new_dataset_dict = dict()
    for k, v in ner_datasets.items():
        tokens = []
        ner_tags = []
        doc_id = []
        snt_id = []
        bos_ids = []
        poss = []
        for doc in v:
            bos = doc["bos_ids"]
            for _snt_id, (s, e) in enumerate(zip(bos, bos[1:] + [len(doc["tokens"])])):
                tokens.append(doc["tokens"][s:e])
                ner_tags.append(doc["ner_tags"][s:e])
                poss.append(doc["POS"][s:e])
                doc_id.append(doc["doc_id"])
                snt_id.append(_snt_id)
                bos_ids.append([0])
        new_dataset_dict[k] = Dataset.from_dict(
            {
                "tokens": tokens,
                "ner_tags": ner_tags,
                "doc_id": doc_id,
                "snt_id": snt_id,
                "bos_ids": bos_ids,
                "POS": poss,
            },
            features=ner_features,
        )
    return DatasetDict(new_dataset_dict)


@colt.register("OrigDataset")
class OrigDataset:
    def __new__(
        self,
        task: str,
        pseudo_dataset=None,
        tokenizer_model_name: str = "bert-base-uncased",
        max_length: int = 510,
        debug_mode: bool = False,
        snt_tokenize: bool = False,
    ) -> None:
        assert task in {"JNLPBA", "Twitter"}
        if task == "JNLPBA":
            download_jnlpba_dataset()
            init_genia_tagger()
            orig_dataset = load_document_wise_jnlpba_dataset()
        elif task == "Twitter":
            download_twitter_dataset()
            twitter_buffer = "data/buffer/Twitter.pkl"
            if not os.path.exists(twitter_buffer):
                orig_dataset = load_twitter_dataset()
                orig_dataset = add_pos(orig_dataset)
                with open(twitter_buffer, "wb") as f:
                    pickle.dump(orig_dataset, f)
            else:
                with open(twitter_buffer, "rb") as f:
                    orig_dataset = pickle.load(f)

        else:
            raise NotImplementedError
        info = datasets.DatasetInfo(
            features=get_ner_dataset_features(task_name2ner_label_names[task])
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name, use_fast=True, add_prefix_space=True
        )
        new_datasets = dict()
        for key in ["train", "validation", "test"]:
            dataset = orig_dataset[key]
            tokens = dataset["tokens"]
            ner_tags = [
                [
                    [info.features["ner_tags"].feature.names.index(l) for l in snt]
                    for snt in doc
                ]
                for doc in dataset["ner_tags"]
            ]
            new_tokens, new_ner_tags, doc_ids, boss, new_poss = (
                [],
                [],
                [],
                [],
                [],
            )  # boss: begin of sentences
            assert len(tokens) == len(ner_tags)
            for doc_tok, doc_ner_tags, doc_poss in zip(
                tokens, ner_tags, dataset["poss"]
            ):
                bos = []
                pos = []
                wide_token_seq, tokenized_wide_token_seq = [], []
                wide_ner_seq = []
                assert len(doc_tok) == len(doc_ner_tags)
                for snt, tags, _pos in zip(doc_tok, doc_ner_tags, doc_poss):
                    tokenized_snt = tokenizer.tokenize(snt, is_split_into_words=True)
                    assert len(tokenized_wide_token_seq) < max_length
                    if len(tokenized_wide_token_seq) + len(tokenized_snt) < max_length:
                        bos.append(len(wide_token_seq))
                        pos += _pos
                        wide_token_seq += snt
                        tokenized_wide_token_seq += tokenized_snt
                        wide_ner_seq += tags
                    else:
                        new_tokens.append(wide_token_seq)
                        new_ner_tags.append(wide_ner_seq)
                        boss.append(bos)
                        new_poss.append(pos)
                        pos = []
                        bos = [0]
                        wide_token_seq = snt
                        tokenized_wide_token_seq = tokenized_snt
                        wide_ner_seq = tags
                new_tokens.append(wide_token_seq)
                new_ner_tags.append(wide_ner_seq)
                boss.append(bos)
                new_poss.append(pos)
            assert [len(tok) for tok in new_tokens] == [len(nt) for nt in new_ner_tags]
            tokens = new_tokens
            ner_tags = new_ner_tags
            doc_ids = list(range(len(tokens)))
            snt_ids = [-1] * len(doc_ids)
            tokens, ner_tags = remove_space_token(tokens, ner_tags)
            new_datasets[key] = Dataset.from_dict(
                {
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                    "bos_ids": boss,
                    "doc_id": doc_ids,
                    "snt_id": snt_ids,
                    "POS": new_poss,
                },
                info=info,
            )
        orig_dataset = DatasetDict(new_datasets)
        if pseudo_dataset:
            assert isinstance(pseudo_dataset.datasets, DatasetDict)
            pd = pseudo_dataset.datasets
            # translate ner_tags
            sup_ner_labels = new_datasets["test"].features["ner_tags"].feature.names
            pseudo_ner_labels = pd["train"].features["ner_tags"].feature.names
            map_sup_to_pseudo = {i: l for i, l in enumerate(sup_ner_labels)}
            new_dataset_dict = dict()
            features = get_ner_dataset_features(pseudo_ner_labels)
            for key, split in new_datasets.items():
                if key in {"validation", "test"}:
                    new_dataset = defaultdict(list)
                    for feature in split.features:
                        if "ner_tags" == feature:
                            new_dataset[feature] = [
                                [map_sup_to_pseudo[w] for w in snt]
                                for snt in split[feature]
                            ]
                        else:
                            new_dataset[feature] = split[feature]
                    new_dataset_dict[key] = Dataset.from_dict(
                        new_dataset, features=features
                    )
            dataset = DatasetDict(
                {
                    "train": pd["train"],
                    # "validation": pd["validation"],
                    "validation": new_dataset_dict["validation"],
                    "supervised_validation": new_dataset_dict["validation"],
                    "test": new_dataset_dict["test"],
                }
            )
        else:
            dataset = orig_dataset
        if debug_mode:
            new_dataset = dict()
            for key, split in dataset.items():
                new_dataset[key] = Dataset.from_dict(split[:10], info=info)
            dataset = DatasetDict(new_dataset)
        if snt_tokenize:
            dataset = snt_tokenize_ner_datasets(dataset)
        return dataset
