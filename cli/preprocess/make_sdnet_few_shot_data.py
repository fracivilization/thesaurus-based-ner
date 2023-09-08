import click
from datasets import DatasetDict, Dataset
from src.dataset.few_shot_sample import few_sample_from_dataset
from src.dataset.utils import split_ner_dataset, translate_label_name_into_common_noun
from typing import Dict
import os
import json
from seqeval.metrics.sequence_labeling import get_entities

SHOT_NUMS = [5, 10, 20, 40, 80, 160, 320, 640]


def translate_conll_format_snt_into_sdnet_format(
    conll_formatted_ner_dataset: Dict,
) -> Dict:
    """conll formatの文をsdnet formatに変換する"""
    tokens = conll_formatted_ner_dataset["tokens"]
    entities = []
    for l, s, e in get_entities(conll_formatted_ner_dataset["ner_tags"]):
        entities.append(
            {"text": " ".join(tokens[s : e + 1]), "offset": [s, e + 1], "type": l}
        )
    return {"tokens": conll_formatted_ner_dataset["tokens"], "entity": entities}


@click.command()
@click.option("--source-datasetdict", type=str, required=True)
@click.option("--output-dir", type=str, required=True)
def main(source_datasetdict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    source_datasetdict = DatasetDict.load_from_disk(source_datasetdict)

    train_datasets = split_ner_dataset(source_datasetdict["train"], split_num=10)

    for few_shot_num in SHOT_NUMS:
        with open(os.path.join(output_dir, f"{few_shot_num}shot.json"), "w") as f:
            for train_dataset in train_datasets:
                few_shot_train_dataset = few_sample_from_dataset(
                    train_dataset, few_shot_num
                )
                tag_names = few_shot_train_dataset.features["ner_tags"].feature.names
                sdnet_formatted_train_few_shot_data = {
                    "support": [],
                    "target_label": list(set(l for l, _, _ in get_entities(tag_names))),
                }
                for snt in few_shot_train_dataset:
                    sdnet_formatted_train_few_shot_data["support"].append(
                        translate_conll_format_snt_into_sdnet_format(
                            {
                                "tokens": snt["tokens"],
                                "ner_tags": [tag_names[tag] for tag in snt["ner_tags"]],
                            }
                        )
                    )
                f.write(json.dumps(sdnet_formatted_train_few_shot_data) + "\n")

    # テストデータをsdnetの形式で出力する
    test_tag_names = source_datasetdict["test"].features["ner_tags"].feature.names
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        for snt in source_datasetdict["test"]:
            sdnet_format_snt = translate_conll_format_snt_into_sdnet_format(
                {
                    "tokens": snt["tokens"],
                    "ner_tags": [test_tag_names[tag] for tag in snt["ner_tags"]],
                }
            )
            f.write(json.dumps(sdnet_format_snt) + "\n")

    # TODO: ラベル名とその説明を出力する
    ## TODO: testデータからラベル名一覧を取ってくる
    label_names = list(set(l for l, _, _ in get_entities(test_tag_names)))
    mapping = {
        label_name: translate_label_name_into_common_noun(label_name)
        for label_name in label_names
    }
    with open(os.path.join(output_dir, "mapping.json"), "w") as f:
        json.dump(mapping, f)


if __name__ == "__main__":
    main()
