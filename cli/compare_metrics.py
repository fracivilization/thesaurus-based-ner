#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import List
from mlflow.tracking import MlflowClient
import click

from prettytable import PrettyTable


def show_metrics(metric_keys: List[str], base_metrics, focus_metrics, metric2delta):
    tbl = PrettyTable([""] + metric_keys)
    tbl.add_row(["Base"] + [base_metrics[k] for k in metric_keys])
    tbl.add_row(["Focus"] + [focus_metrics[k] for k in metric_keys])
    tbl.add_row(["Delta"] + [metric2delta[k] for k in metric_keys])
    print(tbl.get_string())


import click


@click.command()
@click.option("--base-run-id")
@click.option("--focus-run-id")
def cmd(base_run_id, focus_run_id):
    client = MlflowClient()
    base_run = client.get_run(base_run_id)
    base_metrics = base_run.data.metrics
    len(base_metrics)
    print()

    focus_run = client.get_run(focus_run_id)
    focus_metrics = focus_run.data.metrics
    print(focus_run.data.metrics)

    metric2delta = dict()
    assert set(base_metrics.keys()) == set(focus_metrics.keys())
    for key in base_run.data.metrics:
        metric2delta[key] = focus_metrics[key] - base_metrics[key]

    print("Strict P./R./F.")
    show_metrics(
        ["precision", "recall", "f1"], base_metrics, focus_metrics, metric2delta
    )
    print("Lenient P./R./F.")
    show_metrics(
        ["lenient P.", "lenient R.", "lenient F."],
        base_metrics,
        focus_metrics,
        metric2delta,
    )


def main():
    cmd()


if __name__ == "__main__":
    main()
