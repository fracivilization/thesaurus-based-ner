include ./makefiles/__init__.mk

make_dict: $(DICT_FILES)
	@echo DICT_FILES: $(DICT_FILES)
make_gold_msmlc: $(GOLD_TRAIN_MSMLC_DATA)
	@echo GOLD_TRAIN_MSMLC_DATA: $(GOLD_TRAIN_MSMLC_DATA)
make_gold_multi_label_ner: $(GOLD_MULTI_LABEL_NER_DATA)
	@echo $(GOLD_MULTI_LABEL_NER_DATA)
make_gold_ner_data: $(GOLD_DATA)
	@echo $(GOLD_DATA)
make_pseudo_data_on_gold: $(PSEUDO_DATA_ON_GOLD)
	@echo $(NEGATIVE_CATS)
	@echo $(TERM2CAT)
	@echo $(PSEUDO_DATA_ARGS)
	@echo $(PSEUDO_DATA_ON_GOLD)
make_pseudo_multi_label_ner_on_gold: $(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD)
	@echo $(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD)
make_term2cats: $(TERM2CATS)
	@echo $(TERM2CATS)
make_term2cat: $(TERM2CAT)
	@echo $(TERM2CAT)

# Train Main (Flat NER) Model
train: $(TRAIN_OUT)
	@echo TRAIN_OUT: $(TRAIN_OUT)
train_on_gold: $(TRAIN_ON_GOLD_OUT) 
	@echo TRAIN_ON_GOLD_OUT: $(TRAIN_ON_GOLD_OUT)
train_pseudo_anno: $(PSEUDO_OUT)
	@echo $(PSEUDO_OUT)
train_msmlc: $(GOLD_TRAINED_MSMLC_MODEL)
	@echo $(GOLD_TRAINED_MSMLC_MODEL)

# Check Pseudo MSMLC
check_pseudo_msmlc: $(GOLD_TRAIN_MSMLC_DATA) $(TERM2CATS)
	poetry run python -m cli.train_msmlc +multi_label_typer=MultiLabelDictMatchTyper ++msmlc_datasets=$(GOLD_TRAIN_MSMLC_DATA) multi_label_typer.term2cats=$(TERM2CATS)

# Eval flatten marginal_softmax
eval_flatten_marginal_softmax_gold: $(EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT)
	@echo EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT: $(EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT)
eval_flatten_marginal_softmax: $(EVAL_FLATTEN_MARGINAL_MSMLC_OUT)