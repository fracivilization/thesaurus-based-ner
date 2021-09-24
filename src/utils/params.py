from typing import List, Tuple
import datasets

pseudo_annotated_time = 1619178175
Word = str
Tokens = List[str]
TokenBasedSpan = Tuple[int, int]
# e.g. for "I love MacDonald's .", "I" equals (0,1) as token based span
Label = List[str]
CharBasedSpan = Tuple[int, int]
Snts = List[str]  # each sentence is joint of words e.g. "I love MacDonald's ."
LabeledSnts = List[Tuple["str", "str"]]
token_feature = datasets.Sequence(datasets.Value("string"))
token_info = datasets.DatasetInfo(features=datasets.Features({"tokens": token_feature}))

task_name2ner_label_names = {
    "JNLPBA": [
        "O",
        "B-DNA",
        "I-DNA",
        "B-RNA",
        "I-RNA",
        "B-protein",
        "I-protein",
        "B-cell_type",
        "I-cell_type",
        "B-cell_line",
        "I-cell_line",
    ],
    "Twitter": [
        "O",
        "B-person",
        "I-person",
        "B-company",
        "I-company",
        "B-product",
        "I-product",
        "B-facility",
        "I-facility",
        "B-geo-loc",
        "I-geo-loc",
        "B-tvshow",
        "I-tvshow",
        "B-sportsteam",
        "I-sportsteam",
        "B-musicartist",
        "I-musicartist",
        "B-movie",
        "I-movie",
        "B-other",
        "I-other",
    ],
}
task_name2label_names = {"JNLPBA": ["DNA", "RNA", "protein", "cell_type", "cell_line"]}
genia_names = ["DNA", "RNA", "protein", "cell_type", "cell_line"]


unlabelled_corpus_dataset_features = datasets.Features(
    {
        "tokens": datasets.Sequence(datasets.Value("string")),
        "bos_ids": datasets.Sequence(
            datasets.Value("int32")
        ),  # bos: begin of sentence, if there are only one sentence, then "bos_ids" become [0]
        "doc_id": datasets.Value("int32"),
        "snt_id": datasets.Value(
            "int32"
        ),  # if snt_id == 1, it means that it is a not sentence but a document .
        "POS": datasets.Sequence(datasets.Value("string")),
    }
)


def get_ner_labels(focused_types: List[str]):
    ret = ["O"]
    for t in set(focused_types):
        ret.append("B-%s" % t)
        ret.append("I-%s" % t)
    return ret


def get_ner_dataset_features(ner_labels: list) -> datasets.Features:
    assert all(l == "O" or l[:2] in {"B-", "I-", "E-", "S-"} for l in ner_labels)
    features = datasets.Features(
        {
            "tokens": datasets.Sequence(datasets.Value("string")),
            "ner_tags": datasets.Sequence(datasets.ClassLabel(names=ner_labels)),
        }
    )
    return features
