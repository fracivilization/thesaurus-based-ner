include ./makefiles/__init__.mk




test: 
	@echo MSC_O_SAMPLING_RATIO: $(MSC_O_SAMPLING_RATIO)
	@echo MSC_ARGS: $(MSC_ARGS)
	@echo UMLS_CATS: $(UMLS_CATS)
	@echo RAW_CORPUS_OUT: $(RAW_CORPUS_OUT)
	@echo PSEUDO_DATA_BASE_CMD $(PSEUDO_DATA_BASE_CMD)
term2cat: $(TERM2CAT)
	@echo TERM2CAT: $(TERM2CAT)

all: ${DICT_FILES} $(PSEUDO_NER_DATA_DIR) $(PSEUDO_MSC_NER_DATA_DIR) $(GOLD_DATA) $(GOLD_MSC_DATA) $(PSEUDO_DATA_ON_GOLD) $(PSEUDO_MSC_DATA_ON_GOLD) $(FP_REMOVED_PSEUDO_DATA)
train: $(TRAIN_OUT)
	@echo TRAIN_OUT: $(TRAIN_OUT)
train_on_gold: $(TRAIN_ON_GOLD_OUT) 
	@echo TRAIN_ON_GOLD_OUT: $(TRAIN_ON_GOLD_OUT)
train_pseudo_anno: $(PSEUDO_OUT)
	@echo $(PSEUDO_OUT)
make_gold_binary_msmlc: $(GOLD_MSMLC_BINARY_DATA)
	echo $(GOLD_MSMLC_BINARY_DATA)
make_gold_msmlc: $(GOLD_MSMLC_DATA)
	echo $(GOLD_MSMLC_DATA)
make_pseudo_multi_label_ner: $(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD)
make_pseudo_msmlc: $(PSEUDO_MSMLC_DATA_ON_GOLD)
make_umls_term2cats: $(UMLS_TERM2CATS)
	@echo UMLS_TERM2CATS: $(UMLS_TERM2CATS)

check_pseudo_msmlc: $(GOLD_MSMLC_DATA) $(UMLS_TERM2CATS)
	poetry run python -m cli.train_msmlc +multi_label_typer=MultiLabelDictMatchTyper ++msmlc_datasets=$(GOLD_MSMLC_DATA) multi_label_typer.term2cats=$(UMLS_TERM2CATS)
check_pseudo_msmlc_on_ner: $(UMLS_TERM2CATS)
	poetry run python -m cli.train \ 
		ner_model=PseudoMultiLabelTwoStage \
		ner_model.multi_label_typer.term2cats=$(UMLS_TERM2CATS) \
		ner_model.focus_cats=$(subst $() ,_,$(FOCUS_CATS)) \
		++dataset.name_or_path=$(GOLD_DATA) \
		+testor.baseline_typer.term2cat=$(TERM2CAT) 2>&1 | tee ${PSEUDO_OUT}
train_msmlc: $(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL)
	@echo $(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL)
train_msmlc_gold: $(GOLD_TRAINED_MSMLC_MODEL)
	@echo $(GOLD_TRAINED_MSMLC_MODEL)
train_flattern_multilabel_ner: $(PSEUDO_ON_GOLD_FLATTEN_MULTILABEL_NER_OUTPUT)
	@echo PSEUDO_ON_GOLD_FLATTEN_MULTILABEL_NER_OUTPUT: $(PSEUDO_ON_GOLD_FLATTEN_MULTILABEL_NER_OUTPUT)
eval_flatten_marginal_softmax_gold: $(EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT)
eval_flatten_marginal_softmax: $(EVAL_FLATTEN_MARGINAL_MSMLC_OUT)
make_umls_dict_files: $(UMLS_DICT_FILES)