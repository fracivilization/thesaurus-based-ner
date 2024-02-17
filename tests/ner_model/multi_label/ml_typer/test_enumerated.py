import unittest
from src.ner_model.multi_label.ml_typer import (
    MultiLabelEnumeratedTyper,
    MultiLabelEnumeratedTyperConfig,
)
from src.ner_model.multi_label.ml_typer.enumerated import MultiLabelEnumeratedModelArguments, MultiLabelEnumeratedDataTrainingArguments
from datasets import DatasetDict
from src.utils.hydra import HydraAddaptedTrainingArguments

CONLL_POSITIVE_CATS="PER LOC ORG MISC"
CONLL_NEGATIVE_CATS="<http://dbpedia.org/ontology/Activity> <http://dbpedia.org/ontology/Algorithm> <http://dbpedia.org/ontology/Altitude> <http://dbpedia.org/ontology/Amphibian> <http://dbpedia.org/ontology/AnatomicalStructure> <http://dbpedia.org/ontology/Arachnid> <http://dbpedia.org/ontology/Archaea> <http://dbpedia.org/ontology/ArchitecturalStructure> <http://dbpedia.org/ontology/Area> <http://dbpedia.org/ontology/Bacteria> <http://dbpedia.org/ontology/Biomolecule> <http://dbpedia.org/ontology/Bird> <http://dbpedia.org/ontology/Blazon> <http://dbpedia.org/ontology/Browser> <http://dbpedia.org/ontology/ChartsPlacements> <http://dbpedia.org/ontology/ChemicalSubstance> <http://dbpedia.org/ontology/Cipher> <http://dbpedia.org/ontology/Colour> <http://dbpedia.org/ontology/Covid19> <http://dbpedia.org/ontology/Crustacean> <http://dbpedia.org/ontology/Currency> <http://dbpedia.org/ontology/Deity> <http://dbpedia.org/ontology/Demographics> <http://dbpedia.org/ontology/Depth> <http://dbpedia.org/ontology/Diploma> <http://dbpedia.org/ontology/ElectionDiagram> <http://dbpedia.org/ontology/Employer> <http://dbpedia.org/ontology/Family> <http://dbpedia.org/ontology/FictionalCharacter> <http://dbpedia.org/ontology/FileSystem> <http://dbpedia.org/ontology/Fish> <http://dbpedia.org/ontology/Flag> <http://dbpedia.org/ontology/Food> <http://dbpedia.org/ontology/Fungus> <http://dbpedia.org/ontology/GeneLocation> <http://dbpedia.org/ontology/GrossDomesticProduct> <http://dbpedia.org/ontology/GrossDomesticProductPerCapita> <http://dbpedia.org/ontology/Identifier> <http://dbpedia.org/ontology/Insect> <http://dbpedia.org/ontology/Language> <http://dbpedia.org/ontology/List> <http://dbpedia.org/ontology/Mammal> <http://dbpedia.org/ontology/Media> <http://dbpedia.org/ontology/MedicalSpecialty> <http://dbpedia.org/ontology/Medicine> <http://dbpedia.org/ontology/Mollusca> <http://dbpedia.org/ontology/Pandemic> <http://dbpedia.org/ontology/PersonFunction> <http://dbpedia.org/ontology/Plant> <http://dbpedia.org/ontology/Population> <http://dbpedia.org/ontology/Protocol> <http://dbpedia.org/ontology/PublicService> <http://dbpedia.org/ontology/Relationship> <http://dbpedia.org/ontology/Reptile> <http://dbpedia.org/ontology/Skos> <http://dbpedia.org/ontology/SportCompetitionResult> <http://dbpedia.org/ontology/SportsSeason> <http://dbpedia.org/ontology/Spreadsheet> <http://dbpedia.org/ontology/StarCluster> <http://dbpedia.org/ontology/Statistic> <http://dbpedia.org/ontology/Tank> <http://dbpedia.org/ontology/TimePeriod> <http://dbpedia.org/ontology/TopicalConcept> <http://dbpedia.org/ontology/UnitOfWork> <http://dbpedia.org/ontology/Unknown> <http://dbpedia.org/ontology/prov:Entity> <http://dbpedia.org/ontology/prov:Revision> <http://dbpedia.org/ontology/نَسبہ> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> <http://www.w3.org/2002/07/owl#DatatypeProperty> <http://www.w3.org/2002/07/owl#ObjectProperty>"
NER_DATASET_PATH = "tests/fixtures/mini_conll_ner_dataset"
MSMLC_DATASET_PATH = "tests/fixtures/mini_conll_msmlc_dataset"
PSEUDO_CONLL_MSMLC_DATASET_PATH = "tests/fixtures/mini_pseudo_conll_msmlc_dataset"
PSEUDO_MEDMENTIONS_MSMLC_DATASET_PATH = "tests/fixtures/mini_pseudo_medmentions_msmlc_dataset"


class TestEnumeratedTyper(unittest.TestCase):
    def test_load_enumerated_typer(self):
        config = MultiLabelEnumeratedTyperConfig(train_msmlc_datasets=MSMLC_DATASET_PATH)
        MultiLabelEnumeratedTyper(config)

    def test_predict_enumerated_typer(self):
        msmlc_dataset = DatasetDict.load_from_disk(MSMLC_DATASET_PATH)
        config = MultiLabelEnumeratedTyperConfig(train_msmlc_datasets=MSMLC_DATASET_PATH)
        ml_typer = MultiLabelEnumeratedTyper(config)
        validation_dataset = msmlc_dataset["validation"]
        tokens, starts, ends = (
            validation_dataset["tokens"],
            validation_dataset["starts"],
            validation_dataset["ends"],
        )
        ml_typer.batch_predict(tokens, starts, ends)

    def test_train_enumerated_typer(self):
        config = MultiLabelEnumeratedTyperConfig(train_msmlc_datasets=MSMLC_DATASET_PATH)
        ml_typer = MultiLabelEnumeratedTyper(config)
        ml_typer.train()

    def test_train_euumerated_typer_with_16bit(self):
        # train_args で 16bitを指定
        train_args = HydraAddaptedTrainingArguments(fp16=True, output_dir=".")
        config = MultiLabelEnumeratedTyperConfig(
            train_msmlc_datasets=MSMLC_DATASET_PATH, train_args=train_args
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()

    def test_train_euumerated_typer_with_early_stopping(self):
        # c.f. https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=20,
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
        config = MultiLabelEnumeratedTyperConfig(
            train_msmlc_datasets=MSMLC_DATASET_PATH, 
            validation_ner_datasets=NER_DATASET_PATH,
            train_args=train_args,
            model_args=model_args,
            model_output_path="data/model/trained_msmlc_model"
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()

    def test_pseudo_conll_with_early_stopping(self):
        # c.f. https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/
        data_args = MultiLabelEnumeratedDataTrainingArguments(
            positive_cats=CONLL_POSITIVE_CATS,
            negative_cats=CONLL_NEGATIVE_CATS
        )
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            overwrite_output_dir=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        model_args = MultiLabelEnumeratedModelArguments(
            model_name_or_path='bert-base-cased',
            loss_func="MarginalCrossEntropyLoss",
            dynamic_pn_ratio_equivalence=False,
            static_pn_ratio_equivalence=False,
        )
        config = MultiLabelEnumeratedTyperConfig(
            train_msmlc_datasets=PSEUDO_CONLL_MSMLC_DATASET_PATH, 
            validation_ner_datasets=NER_DATASET_PATH,
            train_args=train_args,
            data_args=data_args,
            model_output_path="data/model/trained_msmlc_model",
            model_args=model_args
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()

    def test_pseudo_medmentios_with_early_stopping(self):
        # c.f. https://dev.classmethod.jp/articles/huggingface-usage-early-stopping/
        train_args = HydraAddaptedTrainingArguments(
            output_dir="tmp",
            load_best_model_at_end=True,
            num_train_epochs=20,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            do_train=True,
            overwrite_output_dir=True,
            metric_for_best_model="f1",
            greater_is_better=True,
        )
        model_args = MultiLabelEnumeratedModelArguments(
            model_name_or_path='bert-base-cased',
            loss_func="MarginalCrossEntropyLoss",
            dynamic_pn_ratio_equivalence=False,
            static_pn_ratio_equivalence=False,
        )
        config = MultiLabelEnumeratedTyperConfig(
            train_msmlc_datasets=PSEUDO_MEDMENTIONS_MSMLC_DATASET_PATH, train_args=train_args,
            model_output_path="data/model/trained_msmlc_model"
        )
        typer = MultiLabelEnumeratedTyper(config)
        typer.train()
