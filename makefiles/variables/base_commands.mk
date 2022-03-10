PSEUDO_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_pseudo_ner \
		++ner_model.typer.term2cat=$(TERM2CAT) \
        +gold_corpus=$(GOLD_DATA)
MSC_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_msc_dataset chunker=$(FIRST_STAGE_CHUNKER) ++with_o=$(WITH_O) ++o_sampling_ratio=$(MSC_O_SAMPLING_RATIO)
MSMLC_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_msmlc_dataset +chunker=$(FIRST_STAGE_CHUNKER) ++under_sample=$(UNDERSAMPLE_MSLC) ++with_o=True
MSMLC_BINARY_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_msmlc_dataset +chunker=$(FIRST_STAGE_CHUNKER) ++under_sample=$(UNDERSAMPLE_MSLC) ++with_o=False
FLATTEN_MULTILABEL_NER_BASE_CMD := poetry run python -m cli.train ner_model=flatten_ner \
		ner_model.focus_cats=$(subst $() ,_,$(FOCUS_CATS)) \
		ner_model/multi_label_ner_model=two_stage \
		+ner_model/multi_label_ner_model/chunker=$(FIRST_STAGE_CHUNKER) \
		+ner_model/multi_label_ner_model/multi_label_typer=enumerated \
		++ner_model.multi_label_ner_model.multi_label_typer.prediction_threshold=$(FLATTEN_NER_THRESHOLD) \
		testor.baseline_typer.term2cat=data/term2cat/8a23fbd2bc56b5182ab677063f52af0497d1d5c6.pkl
FLATTEN_MARGINAL_SOFTMAX_NER_BASE_CMD := poetry run python -m cli.train \
		ner_model=flatten_marginal_softmax_ner \
		ner_model.focus_cats=$(subst $() ,_,$(FOCUS_CATS)) \
		ner_model.negative_cats=$(subst $() ,_,$(NEGATIVE_CATS)) \
		ner_model/multi_label_ner_model=two_stage \
		+ner_model/multi_label_ner_model/chunker=$(FIRST_STAGE_CHUNKER) \
		+ner_model/multi_label_ner_model/multi_label_typer=enumerated \
		testor.baseline_typer.term2cat=$(TERM2CAT) \
		++ner_model.multi_label_ner_model.multi_label_typer.train_args.do_train=False \
		++ner_model.multi_label_ner_model.multi_label_typer.model_output_path="no_output" \
		++msmlc_datasets=$(GOLD_MSMLC_DATA) \
		++ner_model.multi_label_ner_model.multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss
TRAIN_MSMLC_BASE_CMD := poetry run python -m cli.train_msmlc +multi_label_typer=enumerated \
		++multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss \
		++multi_label_typer.model_args.dynamic_pn_ratio_equivalence=$(MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE) \
		++multi_label_typer.model_args.pn_ratio_equivalence=$(MSMLC_PN_RATIO_EQUIVALENCE) \
		++multi_label_typer.model_args.negative_ratio_over_positive=$(MSMLC_NEGATIVE_RATIO_OVER_POSITIVE)


TRAIN_BASE_CMD := poetry run python -m cli.train \
		++dataset.name_or_path=$(GOLD_DATA) \
		ner_model/chunker=$(FIRST_STAGE_CHUNKER) \
		ner_model.typer.model_args.o_sampling_ratio=$(O_SAMPLING_RATIO) \
		ner_model.typer.train_args.per_device_train_batch_size=8 \
		ner_model.typer.train_args.per_device_eval_batch_size=16 \
		ner_model.typer.train_args.do_train=True \
		ner_model.typer.train_args.overwrite_output_dir=True \
		testor.baseline_typer.term2cat=$(TERM2CAT)