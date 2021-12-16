from collections import Counter
from datasets.arrow_dataset import Dataset
import os


class MSCDatasetComparer:
    def __init__(
        self, base_dataset: Dataset, focus_dataset: Dataset, output_dir: str
    ) -> None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        i = 1
        base_dataset[i], focus_dataset[i]
        # over predicted spans を 集計する
        base_label_names = base_dataset.features["labels"].feature.names
        focus_label_names = focus_dataset.features["labels"].feature.names
        over_predicted_spans = []
        under_predicted_spans = []
        base_snts = set([" ".join(snt["tokens"]) for snt in base_dataset])
        focus_snts = set([" ".join(snt["tokens"]) for snt in focus_dataset])
        duplicate_snts = base_snts & focus_snts
        duplicate_base_datasets = [
            snt for snt in base_dataset if " ".join(snt["tokens"]) in duplicate_snts
        ]
        duplicate_focus_datasets = [
            snt for snt in focus_dataset if " ".join(snt["tokens"]) in duplicate_snts
        ]
        for base_snt, focus_snt in zip(
            duplicate_base_datasets, duplicate_focus_datasets
        ):
            assert base_snt["tokens"] == focus_snt["tokens"]
            tokens = base_snt["tokens"]
            snt = " ".join(tokens)
            base_labels = [base_label_names[l] for l in base_snt["labels"]]
            focus_labels = [focus_label_names[l] for l in focus_snt["labels"]]
            base_spans = set(zip(base_snt["starts"], base_snt["ends"], base_labels))
            focus_spans = set(zip(focus_snt["starts"], focus_snt["ends"], focus_labels))
            for s, e, l in focus_spans - base_spans:
                over_predicted_spans.append(
                    (" ".join(tokens[s:e]), l, str(s), str(e), snt)
                )
            for s, e, l in base_spans - focus_spans:
                under_predicted_spans.append(
                    (" ".join(tokens[s:e]), l, str(s), str(e), snt)
                )
        print(
            "Over predicted span labels: ",
            Counter([span[1] for span in over_predicted_spans]),
        )
        print("Over predicted span top 300 ranking")
        print("WORD\tCOUNT")
        print(
            "\n".join(
                [
                    "\t".join(map(str, word))
                    for word in Counter(
                        [
                            span[0].split(" ")[-1].lower()
                            for span in over_predicted_spans
                        ]
                    ).most_common()[:300]
                ]
            )
        )
        with open(os.path.join(output_dir, "over_predicted_spans.tsv"), "w") as f:
            f.write("\n".join(["\t".join(span) for span in over_predicted_spans]))
        print(
            "Under predicted span labels: ",
            Counter([span[1] for span in under_predicted_spans]),
        )
        print("Under predicted span top 300 ranking")
        print("WORD\tCOUNT")
        print(
            "\n".join(
                [
                    "\t".join(map(str, word))
                    for word in Counter(
                        [
                            span[0].split(" ")[-1].lower()
                            for span in under_predicted_spans
                        ]
                    ).most_common()[:300]
                ]
            )
        )
        with open(os.path.join(output_dir, "under_predicted_spans.tsv"), "w") as f:
            f.write("\n".join(["\t".join(span) for span in under_predicted_spans]))
