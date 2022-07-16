from typing import Dict
from datasets import DatasetDict
import unittest
from collections import Counter
from prettytable import PrettyTable
from torch import negative


class TestMSMLCStatistics(unittest.TestCase):
    def test_pnr_print(self):
        gold_msmlc_dataset = DatasetDict.load_from_disk(
            "data/gold/8c00d1c9e813a79bea66498e3fe9ffe89f59ea97"
        )
        negative_ratio_over_positive = 1.0
        test_dataset = gold_msmlc_dataset["train"]
        label_counter = Counter()
        all_span_count = 0
        label_names = test_dataset.features["labels"].feature.feature.names
        for snt in test_dataset:
            labels = snt["labels"]
            for span in labels:
                for label_id in span:
                    label = label_names[label_id]
                    all_span_count += 1
                    label_counter[label] += 1
        all_span_count = label_counter["T000"] * (negative_ratio_over_positive + 1)
        all_positive_count = label_counter["T000"]
        tbl = PrettyTable()
        tbl.field_names = [
            "label",
            "count",
            "PNR (%)",
            "LAPR (%)",
        ]  # LPR: label to all positive ratio: ラベルの正例全体に対する比率
        for label, count in label_counter.items():
            if label == "nc-O":
                tbl.add_row(
                    [
                        label,
                        count,
                        "%.4f"
                        % (
                            100
                            * negative_ratio_over_positive
                            / (negative_ratio_over_positive + 1),
                        ),
                        negative_ratio_over_positive,
                    ]
                )
            else:
                tbl.add_row(
                    [
                        label,
                        count,
                        "%.4f" % (100 * count / all_span_count,),
                        "%.4f" % (100 * count / all_positive_count),
                    ]
                )
        print(tbl)
        pass


if __name__ == "__main__":
    unittest.main()
