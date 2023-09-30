import unittest
from src.ner_model.marginal_softmax_model import FlattenMarginalSoftmaxNERModelConfig, FlattenMarginalSoftmaxNERModel
from src.ner_model.multi_label.ml_typer.enumerated import MultiLabelEnumeratedTyperConfig, MultiLabelEnumeratedModelArguments, MultiLabelEnumeratedTyper
from src.utils.hydra import HydraAddaptedTrainingArguments
from datasets import DatasetDict
from src.ner_model.chunker.abstract_model import EnumeratedChunkerConfig
from src.ner_model.multi_label.two_stage import MultiLabelTwoStageConfig

MSC_DATASET_PATH = "tests/fixtures/mini_conll_msc_dataset"
NER_DATASET_PATH = "tests/fixtures/mini_conll_ner_dataset"
PSEUDO_CONLL_MSMLC_DATASET_PATH = "tests/fixtures/mini_pseudo_conll_msmlc_dataset"


class TestMarginalSoftmax(unittest.TestCase):
    def test_load_marginal_softmax_model(self):
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=4,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            overwrite_output_dir=True
        )
        model_args = MultiLabelEnumeratedModelArguments(
            model_name_or_path='bert-base-cased',
            loss_func="MarginalCrossEntropyLoss",
            dynamic_pn_ratio_equivalence=False,
            static_pn_ratio_equivalence=False,
        )
        config = FlattenMarginalSoftmaxNERModelConfig(
            multi_label_ner_model=MultiLabelTwoStageConfig(
                chunker=EnumeratedChunkerConfig(),
                multi_label_typer=MultiLabelEnumeratedTyperConfig(
                    train_msmlc_datasets=PSEUDO_CONLL_MSMLC_DATASET_PATH,
                    train_args=train_args,
                    model_args=model_args,
                    model_output_path="data/model/trained_msmlc_model"
                ),
            ),
            positive_cats='PER_LOC_ORG_MISC',
            eval_dataset='CoNLL2003',
            with_negative_categories=False
        )
        ner_model = FlattenMarginalSoftmaxNERModel(config)
        ner_dataset = DatasetDict.load_from_disk(NER_DATASET_PATH)
        output = ner_model.batch_predict(ner_dataset['train']['tokens'])
        print(output)