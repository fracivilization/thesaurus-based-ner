import subprocess

MSC_DATASET = "tests/fixtures/mini_conll_msc_dataset"
MSMLC_DATASET = "tests/fixtures/mini_conll_msmlc_dataset"
NER_DATASET="tests/fixtures/mini_conll_ner_dataset"

class TestTrainMSCEnumerated:
    def test_run_train_msc_enumerated(self):
        # TODO: 100/100/100のテスト用データセットを作る
        cmd = [
            "poetry", "run", "python", "-m",
            "cli.train", 
            f"dataset.name_or_path={NER_DATASET}",
            "ner_model/chunker=enumerated",
            "ner_model.typer.model_args.model_name_or_path='bert-base-cased'",
            "ner_model.typer.model_args.negative_ratio_over_positive=1.0",
            "ner_model.typer.train_args.per_device_train_batch_size=8",
            "ner_model.typer.train_args.per_device_eval_batch_size=16",
            "ner_model.typer.train_args.num_train_epochs=40",
            "ner_model.typer.train_args.do_train=True",
            "ner_model.typer.train_args.overwrite_output_dir=True",
            "ner_model.typer.train_args.save_total_limit=5",
            # 学習の最後に最も良かったモデルを利用する（Early Stoppingに必要）
            "ner_model.typer.train_args.load_best_model_at_end=True",
            # 学習の最後に最も良かったモデルを利用する（Early Stoppingに必要）
            "ner_model.typer.train_args.save_strategy=EPOCH",
            "ner_model.typer.train_args.evaluation_strategy=EPOCH",
            "ner_model.typer.model_args.dynamic_pn_ratio_equivalence=True",
            f"ner_model.typer.msc_datasets={MSC_DATASET}"
        ]
        print(cmd)
        subprocess.run(cmd)


class TestTrainMSMLCEnumerated:
    def test_run_train_msmlc_enumerated(self):
        # TODO: 100/100/100のテスト用データセットを作る
        cmd = [
            "poetry", "run", "python", "-m",
            "cli.train_msmlc",
            "+multi_label_typer=enumerated",
            "++multi_label_typer.model_args.model_name_or_path='bert-base-cased'",
            "++multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss",
            "++multi_label_typer.train_args.num_train_epochs=40",
            "++multi_label_typer.train_args.per_device_train_batch_size=8",
            "++multi_label_typer.train_args.per_device_eval_batch_size=16",
            "++multi_label_typer.train_args.save_total_limit=5",
            "++multi_label_typer.model_args.dynamic_pn_ratio_equivalence=False",
            "++multi_label_typer.model_args.static_pn_ratio_equivalence=False",
            f"++multi_label_typer.train_datasets={MSMLC_DATASET}",
            "++multi_label_typer.model_output_path=data/model/trained_msmlc_model",
            # 学習の最後に最も良かったモデルを利用する（Early Stoppingに必要）
            "++multi_label_typer.train_args.load_best_model_at_end=True",
            # 学習の最後に最も良かったモデルを利用する（Early Stoppingに必要）
            "++multi_label_typer.train_args.save_strategy=EPOCH",
            "++multi_label_typer.train_args.evaluation_strategy=EPOCH",
        ]
        print(cmd)
        subprocess.run(cmd)
class TestTrainAndEvalFlattenSoftmax:
    def test_train_and_eval_flatten_softmax(self):
        cmd = [
            "poetry", "run", "python", "-m",
            "cli.train",
            f"dataset.name_or_path={NER_DATASET}",
            "ner_model=flatten_marginal_softmax_ner",
            "ner_model.positive_cats=PER_LOC_ORG_MISC",
            "ner_model.with_negative_categories=False",
            "ner_model.eval_dataset=CoNLL2003",
            "ner_model/multi_label_ner_model=two_stage",
            "+ner_model/multi_label_ner_model/chunker=enumerated",
            "+ner_model/multi_label_ner_model/multi_label_typer=enumerated",
            "++ner_model.multi_label_ner_model.multi_label_typer.model_args.model_name_or_path='bert-base-cased'",
            "++ner_model.multi_label_ner_model.multi_label_typer.train_args.do_train=True",
            "++ner_model.multi_label_ner_model.multi_label_typer.model_output_path=\"no_output\"",
            f"++msmlc_datasets={MSMLC_DATASET}",
            "++ner_model.multi_label_ner_model.multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss",
            f"++ner_model.multi_label_ner_model.multi_label_typer.train_datasets={MSMLC_DATASET}"
        ]
        print(cmd)
        subprocess.run(cmd)

