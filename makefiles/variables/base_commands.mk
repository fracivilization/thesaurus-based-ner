PSEUDO_DATA_BASE_CMD := ${PYTHON} -m cli.preprocess.load_pseudo_ner \
		ner_model=$(EVAL_DATASET)PseudoTwoStage \
		++ner_model.typer.term2cat=$(TERM2CAT) \
        +gold_corpus=$(GOLD_DATA)
MSC_DATA_BASE_CMD := ${PYTHON} -m cli.preprocess.load_msc_dataset chunker=$(FIRST_STAGE_CHUNKER) ++with_o=$(WITH_O) 
MSMLC_DATA_BASE_CMD := ${PYTHON} -m cli.preprocess.load_msmlc_dataset +chunker=$(FIRST_STAGE_CHUNKER) ++with_o=True
TRAIN_COMMON_BASE_CMD := ${PYTHON} -m cli.train \
		dataset.name_or_path=$(GOLD_DATA)
FLATTEN_MARGINAL_SOFTMAX_NER_BASE_CMD := $(TRAIN_COMMON_BASE_CMD) \
		ner_model=flatten_marginal_softmax_ner \
		ner_model.positive_cats=$(subst $() ,_,$(POSITIVE_CATS)) \
		ner_model.with_negative_categories=$(WITH_NEGATIVE_CATEGORIES) \
		ner_model.eval_dataset=$(EVAL_DATASET) \
		ner_model/multi_label_ner_model=two_stage \
		+ner_model/multi_label_ner_model/chunker=$(FIRST_STAGE_CHUNKER) \
		+ner_model/multi_label_ner_model/multi_label_typer=enumerated \
		++ner_model.multi_label_ner_model.multi_label_typer.model_args.model_name_or_path=$(MODEL_NAME) \
		++ner_model.multi_label_ner_model.multi_label_typer.train_args.do_train=False \
		++ner_model.multi_label_ner_model.multi_label_typer.model_output_path="no_output" \
		++ner_model.multi_label_ner_model.multi_label_typer.data_args.positive_cats=$(subst $() ,_,$(POSITIVE_CATS)) \
		++ner_model.multi_label_ner_model.multi_label_typer.data_args.eval_dataset_for_negative_categories=$(EVAL_DATASET) \
		++ner_model.multi_label_ner_model.multi_label_typer.data_args.with_negative_categories=$(WITH_NEGATIVE_CATEGORIES) \
		++ner_model.multi_label_ner_model.multi_label_typer.validation_ner_datasets=$(GOLD_DATA) \
		++msmlc_datasets=$(GOLD_TRAIN_MSMLC_DATA) \
		++ner_model.multi_label_ner_model.multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss

TRAIN_MSMLC_BASE_CMD := ${PYTHON} -m cli.train_msmlc +multi_label_typer=enumerated \
		++multi_label_typer.validation_ner_datasets=$(GOLD_DATA) \
		++multi_label_typer.model_args.model_name_or_path=$(MODEL_NAME) \
		++multi_label_typer.model_args.loss_func=MarginalCrossEntropyLoss \
		++multi_label_typer.data_args.early_stopping_patience=$(EARLY_STOPPING_PATIENCE) \
		++multi_label_typer.data_args.positive_cats=$(subst $() ,_,$(POSITIVE_CATS)) \
		++multi_label_typer.data_args.eval_dataset_for_negative_categories=$(EVAL_DATASET) \
		++multi_label_typer.data_args.with_negative_categories=$(WITH_NEGATIVE_CATEGORIES) \
		++multi_label_typer.train_args.num_train_epochs=$(NUM_TRAIN_EPOCHS) \
		++multi_label_typer.train_args.per_device_train_batch_size=$(TRAIN_BATCH_SIZE) \
		++multi_label_typer.train_args.per_device_eval_batch_size=$(EVAL_BATCH_SIZE) \
		++multi_label_typer.train_args.load_best_model_at_end=True \
		++multi_label_typer.train_args.metric_for_best_model=f1 \
		++multi_label_typer.train_args.greater_is_better=True \
		++multi_label_typer.train_args.save_strategy=STEPS \
		++multi_label_typer.train_args.evaluation_strategy=STEPS \
		++multi_label_typer.train_args.save_total_limit=40 \
		++multi_label_typer.train_args.eval_steps=$(EVAL_STEPS) \
		++multi_label_typer.train_args.save_steps=$(EVAL_STEPS) \
		++multi_label_typer.train_args.fp16=True


TRAIN_BASE_CMD := $(TRAIN_COMMON_BASE_CMD) \
		ner_model/chunker=$(FIRST_STAGE_CHUNKER) \
		ner_model.typer.validation_ner_datasets=$(GOLD_DATA) \
		ner_model.typer.model_args.model_name_or_path=$(MODEL_NAME) \
		ner_model.typer.model_args.negative_ratio_over_positive=$(NEGATIVE_RATIO_OVER_POSITIVE) \
		ner_model.typer.train_args.per_device_train_batch_size=$(TRAIN_BATCH_SIZE) \
		ner_model.typer.train_args.per_device_eval_batch_size=$(EVAL_BATCH_SIZE) \
		ner_model.typer.train_args.num_train_epochs=$(NUM_TRAIN_EPOCHS) \
		ner_model.typer.train_args.do_train=True \
		ner_model.typer.train_args.overwrite_output_dir=True \
		ner_model.typer.train_args.save_total_limit=1 \
		ner_model.typer.train_args.load_best_model_at_end=True \
		ner_model.typer.train_args.save_strategy=STEPS \
		ner_model.typer.train_args.evaluation_strategy=STEPS \
		ner_model.typer.train_args.metric_for_best_model=f1 \
		ner_model.typer.train_args.greater_is_better=True \
		ner_model.typer.train_args.eval_steps=$(EVAL_STEPS) \
		ner_model.typer.train_args.save_steps=$(EVAL_STEPS) \
		ner_model.typer.train_args.fp16=True
