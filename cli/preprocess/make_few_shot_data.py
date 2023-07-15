import click
from datasets import DatasetDict, Dataset
import numpy as np
from seqeval.metrics.sequence_labeling import get_entities


def few_sample_from_dataset(dataset: Dataset, sample_num: int):
    # ラベル数をデータから取得する
    tag_names = dataset.features["ner_tags"].feature.names
    labels = list(set([l for l, _, _ in get_entities(tag_names)]))
    dataset_label_counts = []
    for ner_tags in dataset["ner_tags"]:
        ner_tags = [tag_names[tag] for tag in ner_tags]
        dataset_label_count = np.zeros(len(labels), dtype=np.int32)
        for l, _, _ in get_entities(ner_tags):
            dataset_label_count[labels.index(l)] += 1
        dataset_label_counts.append(dataset_label_count)
    dataset_label_counts = np.array(dataset_label_counts)

    target_class_counts = np.ones(len(labels)) * sample_num

    target_data_ids = []
    label_count_sort = np.argsort(-dataset_label_counts.sum(axis=1))
    remained_dataset = dataset_label_counts.sum(axis=1) > 0
    remained_dataset = np.logical_and(
        np.all(dataset_label_counts <= target_class_counts, axis=1),
        remained_dataset,
    )
    while np.any(remained_dataset) and np.any(target_class_counts > 0):

        # remained_datasetのうち、最もラベル数が多いサンプルを取得する
        best_sample_id = label_count_sort[remained_dataset[label_count_sort]][0]
        target_data_ids.append(best_sample_id)
        remained_dataset[best_sample_id] = False

        # target_class_count未満のラベル数を持つサンプルをremained_datasetにする
        target_class_counts -= dataset_label_counts[best_sample_id]
        remained_dataset = np.logical_and(
            np.all(dataset_label_counts <= target_class_counts, axis=1),
            remained_dataset,
        )
    return Dataset.from_dict(dataset[target_data_ids], features=dataset.features)


@click.command()
@click.option("--source-datasetdict", type=str, required=True)
@click.option("--few-shot-num", type=int, required=True)
@click.option("--output-dir", type=str, required=True)
def main(source_datasetdict, few_shot_num, output_dir):
    source_datasetdict = DatasetDict.load_from_disk(source_datasetdict)

    # validation, testはそのまま使う
    ret_datasetdict = {
        "train": few_sample_from_dataset(source_datasetdict["train"], few_shot_num),
        "validation": source_datasetdict["validation"],
        "test": source_datasetdict["test"],
    }
    DatasetDict(ret_datasetdict).save_to_disk(output_dir)


if __name__ == "__main__":
    main()
