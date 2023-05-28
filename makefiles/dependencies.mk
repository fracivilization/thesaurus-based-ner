$(DATA_DIR):
	mkdir $(DATA_DIR)
$(UMLS_DIR): $(DATA_DIR)
	@echo "Please Download UMLS2021AA full from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html"
	@echo "You need UMLS Account, so please acces by web browser and mv the file into $(UMLS_DIR)"
	@echo "Plaese refer to README.md"
$(DBPEDIA_DIR): $(DATA_DIR)
	mkdir -p $(DBPEDIA_DIR)
	# wget https://databus.dbpedia.org/ontologies/dbpedia.org/ontology--DEV/2023.02.21-124003/ontology--DEV_type=parsed_sorted.nt # DBPedia Ontlogy
	# # Wikipedia in DBPedia
	# wget https://databus.dbpedia.org/dbpedia/mappings/instance-types/2021.06.01/instance-types_lang=en_specific.ttl.bz2 # Wikipedia Articles Types
	# wget https://databus.dbpedia.org/dbpedia/generic/labels/2021.06.01/labels_lang=en.ttl.bz2 # Wikipedia Article Label
	# wget https://databus.dbpedia.org/dbpedia/mappings/mappingbased-literals/2021.06.01/mappingbased-literals_lang=en.ttl.bz2 ## Literals extracted with mappings 
	# wget https://databus.dbpedia.org/dbpedia/generic/infobox-properties/2021.06.01/infobox-properties_lang=en.ttl.bz2 ## Extracted facts from Wikipedia Infoboxes 
	# wget https://databus.dbpedia.org/dbpedia/generic/redirects/2021.06.01/redirects_lang=en.ttl.bz2	## redirects dataset
	# # Wikidata in DBPedia
	# wget https://databus.dbpedia.org/dbpedia/wikidata/instance-types/2021.06.01/instance-types_specific.ttl.bz2 # Type of Wikidata Instance
	# wget https://databus.dbpedia.org/dbpedia/wikidata/labels/2021.03.01/labels.ttl.bz2 # Wikidata Labels
	# wget https://databus.dbpedia.org/dbpedia/wikidata/ontology-subclassof/2021.02.01/ontology-subclassof.ttl.bz2 # Wikidata SubClassOf
	# wget https://databus.dbpedia.org/dbpedia/wikidata/alias/2021.02.01/alias.ttl.bz2 # Wikidata Alias
	# wget https://databus.dbpedia.org/dbpedia/generic/disambiguations/2022.12.01/disambiguations_lang=en.ttl.bz2
	# bunzip2 *.bz2
	# mv *.ttl $(DBPEDIA_DIR)
	# mv *.nt $(DBPEDIA_DIR)
$(BUFFER_DIR): $(DATA_DIR)
	mkdir -p $(BUFFER_DIR)

$(TERM2CAT_DIR): $(DATA_DIR)
	mkdir -p $(TERM2CAT_DIR)
$(TERM2CATS_DIR): $(DATA_DIR)
	mkdir -p $(TERM2CATS_DIR)
$(TERM2CAT): $(TERM2CAT_DIR) $(TERM2CATS)
	@echo TERM2CAT: $(TERM2CAT)
	${PYTHON} -m cli.preprocess.load_term2cat \
		term2cats=$(TERM2CATS) \
		output=$(TERM2CAT) \
		positive_cats=$(subst $() ,_,$(POSITIVE_CATS)) \
		eval_dataset=$(EVAL_DATASET) \
		with_negative_categories=$(WITH_NEGATIVE_CATEGORIES)

$(PSEUDO_DATA_DIR): $(DATA_DIR)
	echo $(PSEUDO_DATA_DIR)
	mkdir -p $(PSEUDO_DATA_DIR)

$(GOLD_DIR): $(DATA_DIR)
	mkdir -p $(GOLD_DIR)
$(MED_MENTIONS_DIR): $(GOLD_DIR)
	# git clone https://github.com/chanzuckerberg/MedMentions
	# for f in `find MedMentions/ | grep gz`; do gunzip $$f; done
	# mv MedMentions $(GOLD_DIR)/MedMentions
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_trng.txt data/gold/MedMentions/st21pv/data/
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_dev.txt data/gold/MedMentions/st21pv/data/
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_test.txt data/gold/MedMentions/st21pv/data/
$(CONLL2003_DIR): $(GOLD_DIR)
	@echo CONLL2003_DIR: $(CONLL2003_DIR)
	@echo Please download from https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion?resource=download
	mkdir -p $(CONLL2003_DIR)
$(GOLD_DATA): $(GOLD_MULTI_LABEL_NER_DATA)
	@echo "Gold Data"
	@echo GOLD_MULTI_LABEL_NER_DATA: $(GOLD_MULTI_LABEL_NER_DATA)
	@echo PYTHON -m cli.preprocess.load_gold_ner --positive-cats $(subst $() ,_,$(POSITIVE_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_MULTI_LABEL_NER_DATA) --train-snt-num $(TRAIN_SNT_NUM)
	${PYTHON} -m cli.preprocess.load_gold_ner --positive-cats $(subst $() ,_,$(POSITIVE_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_MULTI_LABEL_NER_DATA) --train-snt-num $(TRAIN_SNT_NUM)
$(GOLD_TRAIN_DATA): $(GOLD_MULTI_LABEL_NER_DATA)
	@echo "Gold Data"
	@echo GOLD_MULTI_LABEL_NER_DATA: $(GOLD_MULTI_LABEL_NER_DATA)
	@echo PYTHON -m cli.preprocess.load_gold_ner --positive-cats $(subst $() ,_,$(POSITIVE_CATS)) --negative-cats $(subst $() ,_,$(NEGATIVE_CATS)) --output $(GOLD_TRAIN_DATA) --input-dir $(GOLD_MULTI_LABEL_NER_DATA) --train-snt-num $(TRAIN_SNT_NUM)
	${PYTHON} -m cli.preprocess.load_gold_ner --positive-cats $(subst $() ,_,$(POSITIVE_CATS)) --negative-cats $(subst $() ,_,$(NEGATIVE_CATS)) --output $(GOLD_TRAIN_DATA) --input-dir $(GOLD_MULTI_LABEL_NER_DATA) --train-snt-num $(TRAIN_SNT_NUM)
$(GOLD_TRAIN_MSC_DATA): $(GOLD_TRAIN_DATA)
	@echo GOLD_TRAIN_MSC_DATA_ON_GOLD: $(GOLD_TRAIN_MSC_DATA)
	$(MSC_DATA_BASE_CMD) \
		++negative_sampling=True \
		++negative_ratio_over_positive=$(NEGATIVE_RATIO_OVER_POSITIVE) \
		+ner_dataset=$(GOLD_TRAIN_DATA) \
		+output_dir=$(GOLD_TRAIN_MSC_DATA)



$(RAW_CORPUS_DIR): $(DATA_DIR)
	mkdir -p $(RAW_CORPUS_DIR)
$(PUBMED): $(RAW_CORPUS_DIR)
	# mkdir -p $(PUBMED)
	# for f in `seq -w 1062`; do wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n$$f.xml.gz ; gunzip pubmed21n$$f.xml.gz & done
	# mv pubmed21n*.xml $(PUBMED)
	# for f in `ls $(PUBMED)/pubmed21n*.xml`; do ${PYTHON} -m cli.preprocess.load_pubmed_txt $$f & done
	

$(RAW_CORPUS_OUT): $(SOURCE_TXT_DIR)
	@echo raw sentence num: $(RAW_SENTENCE_NUM)
	@echo raw corpus out dir: $(RAW_CORPUS_OUT)
	${PYTHON} -m cli.preprocess.load_raw_corpus --raw-sentence-num $(RAW_SENTENCE_NUM) --source-txt-dir $(SOURCE_TXT_DIR) --output-dir $(RAW_CORPUS_OUT)

$(PSEUDO_DATA_ON_GOLD): $(GOLD_DATA) $(PSEUDO_DATA_DIR) $(TERM2CAT) $(BUFFER_DIR)
	@echo make pseudo data on Gold dataset for comparison
	@echo make from Gold: $(GOLD_DATA)
	@echo positive categories: $(POSITIVE_CATS)
	@echo negative categories: $(NEGATIVE_CATS)
	@echo PSEUDO_DATA_ON_GOLD: $(PSEUDO_DATA_ON_GOLD)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(PSEUDO_DATA_ON_GOLD)
$(PSEUDO_MSC_DATA_ON_GOLD): $(PSEUDO_DATA_ON_GOLD)
	@echo PSEUDO_MSC_DATA_ON_GOLD: $(PSEUDO_MSC_DATA_ON_GOLD)
	$(MSC_DATA_BASE_CMD) \
		+ner_dataset=$(PSEUDO_DATA_ON_GOLD) \
		+output_dir=$(PSEUDO_MSC_DATA_ON_GOLD)

$(TRAIN_OUT): $(PSEUDO_MSC_DATA_ON_GOLD) $(GOLD_DATA)
	$(TRAIN_BASE_CMD) \
		ner_model.typer.msc_datasets=$(PSEUDO_MSC_DATA_ON_GOLD) 2>&1 | tee $(TRAIN_OUT)

$(TRAIN_ON_GOLD_OUT): $(GOLD_TRAIN_MSC_DATA) $(GOLD_DATA)
	$(TRAIN_BASE_CMD) \
		ner_model.typer.msc_datasets=$(GOLD_TRAIN_MSC_DATA) 2>&1 | tee $(TRAIN_ON_GOLD_OUT)


$(PSEUDO_OUT): $(GOLD_DATA) $(TERM2CAT)
ifeq ($(REMAIN_COMMON_SENSE_FOR_TERM2CATS),False)
	${PYTHON} -m cli.train \
		ner_model=$(EVAL_DATASET)PseudoTwoStage \
		++dataset.name_or_path=$(GOLD_DATA) \
		+ner_model.typer.term2cat=$(TERM2CAT) 2>&1 | tee ${PSEUDO_OUT}
else
		echo "疑似教師による予測の際には着目クラスのみを利用してください"
endif

$(GOLD_DIR)/MedMentions_multi_label_ner: $(MED_MENTIONS_DIR)
	${PYTHON} -m cli.preprocess.load_gold_multi_label_ner --evaluation-data MedMentions --input-dir $(MED_MENTIONS_DIR)/full/data --output-dir $(GOLD_DIR)/MedMentions_multi_label_ner

$(GOLD_DIR)/CoNLL2003_multi_label_ner: $(CONLL2003_DIR)
	${PYTHON} -m cli.preprocess.load_gold_multi_label_ner --evaluation-data CoNLL2003 --input-dir $(CONLL2003_DIR) --output-dir $(GOLD_DIR)/CoNLL2003_multi_label_ner

$(GOLD_TRAIN_MSMLC_DATA): $(GOLD_MULTI_LABEL_NER_DATA)
	$(MSMLC_DATA_BASE_CMD) \
	+multi_label_ner_dataset=$(GOLD_MULTI_LABEL_NER_DATA) \
	+output_dir=$(GOLD_TRAIN_MSMLC_DATA)

$(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD): $(TERM2CATS) $(GOLD_MULTI_LABEL_NER_DATA) $(PSEUDO_DATA_DIR)
	${PYTHON} -m cli.preprocess.load_pseudo_multi_label_ner \
		+multi_label_ner_model/multi_label_typer=$(EVAL_DATASET)DictMatch \
		++multi_label_ner_model.multi_label_typer.term2cats=$(TERM2CATS) \
		++multi_label_ner_model.remove_null_chunk=True \
		+gold_corpus=$(GOLD_MULTI_LABEL_NER_DATA) \
		+raw_corpus=$(GOLD_MULTI_LABEL_NER_DATA) \
		+output_dir=$(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD)
$(PSEUDO_MSMLC_DATA_ON_GOLD): $(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD)
	$(MSMLC_DATA_BASE_CMD) \
	+multi_label_ner_dataset=$(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD) \
	+output_dir=$(PSEUDO_MSMLC_DATA_ON_GOLD)

$(DICTIONARY_FORM_TERM2CATS): $(TERM2CATS_DIR) $(UMLS_DIR)
ifeq ($(wildcard $(DICTIONARY_FORM_TERM2CATS)),)
	@echo DICTIONARY_FORM_TERM2CATS: $(DICTIONARY_FORM_TERM2CATS)
	${PYTHON} -m cli.preprocess.load_dictionary_form_term2cats \
		--knowledge-base=$(KNOWLEDGE_BASE) \
		--remain-common-sense=$(REMAIN_COMMON_SENSE_FOR_TERM2CATS) \
		--output-dir=$(DICTIONARY_FORM_TERM2CATS)
else
	echo "既にDICTIONARY_FORM_TERM2CATSは存在します"
	echo "DICTIONARY_FORM_TERM2CATS: $(DICTIONARY_FORM_TERM2CATS)"
endif

$(TERM2CATS): $(DICTIONARY_FORM_TERM2CATS)
ifeq ($(wildcard $(TERM2CATS)),)
	${PYTHON} -m cli.preprocess.inflect_terms_of_term2cats \
		--dictionary-form-term2cats-dir=$(DICTIONARY_FORM_TERM2CATS) \
		--output-path=$(TERM2CATS)
else
	echo "既にTERM2CATSは存在します"
	echo "TERM2CATS: $(TERM2CATS)"
endif

$(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL): $(PSEUDO_MSMLC_DATA_ON_GOLD)
	$(TRAIN_MSMLC_BASE_CMD) \
		++multi_label_typer.train_datasets=$(PSEUDO_MSMLC_DATA_ON_GOLD) \
		++multi_label_typer.model_output_path=$(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL)

$(GOLD_TRAINED_MSMLC_MODEL): $(GOLD_TRAIN_MSMLC_DATA)
	$(TRAIN_MSMLC_BASE_CMD) \
		++multi_label_typer.train_datasets=$(GOLD_TRAIN_MSMLC_DATA) \
		++multi_label_typer.model_output_path=$(GOLD_TRAINED_MSMLC_MODEL)

$(EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT): $(GOLD_TRAINED_MSMLC_MODEL) $(GOLD_TRAIN_MSMLC_DATA) $(GOLD_DATA)
	$(FLATTEN_MARGINAL_SOFTMAX_NER_BASE_CMD) \
		++ner_model.multi_label_ner_model.multi_label_typer.model_args.saved_param_path=$(GOLD_TRAINED_MSMLC_MODEL) \
		++ner_model.multi_label_ner_model.multi_label_typer.train_datasets=$(GOLD_TRAIN_MSMLC_DATA) \
		2>&1 | tee $(EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT)

$(EVAL_FLATTEN_MARGINAL_MSMLC_OUT): $(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL) $(PSEUDO_MSMLC_DATA_ON_GOLD) $(GOLD_DATA)
	@echo $(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL)
	$(FLATTEN_MARGINAL_SOFTMAX_NER_BASE_CMD) \
		++ner_model.multi_label_ner_model.multi_label_typer.model_args.saved_param_path=$(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL) \
		++ner_model.multi_label_ner_model.multi_label_typer.train_datasets=$(PSEUDO_MSMLC_DATA_ON_GOLD) \
		2>&1 | tee $(EVAL_FLATTEN_MARGINAL_MSMLC_OUT)


