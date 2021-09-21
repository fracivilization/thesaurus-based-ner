DBPEDIA_CATS = Agent ChemicalSubstance GeneLocation Biomolecule Unknown TopicalConcept
UMLS_CATS = T116
FOCUS_CATS := T116 T126
DUPLICATE_CATS := ChemicalSubstance GeneLocation Biomolecule Unknown TopicalConcept
REMOVE_CATS := $(filter-out $(DUPLICATE_CATS), $(DBPEDIA_CATS))
APPEARED_CATS := $(FOCUS_CATS) $(REMOVE_CATS)
DATA_DIR := data
DICT_DIR := $(DATA_DIR)/dict
DICT_FILES := $(addprefix $(DICT_DIR)/,$(APPEARED_CATS))
RAW_CORPUS_NUM := 50000
# APPEARED_CATS を使って 出力先のフォルダを決める
RAW_CORPUS_DIR := $(DATA_DIR)/raw
PUBMED := $(RAW_CORPUS_DIR)/pubmed
SOURCE_TXT_DIR := $(PUBMED)
RAW_CORPUS_OUT := $(RAW_CORPUS_DIR)/$(firstword $(shell echo $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_DATA_DIR := $(DATA_DIR)/pseudo
PSEUDO_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(APPEARED_CATS) $(RAW_CORPUS_NUM) | sha1sum))

GOLD_DIR := $(DATA_DIR)/gold
GOLD_DATA := $(GOLD_DIR)/$(firstword $(shell echo "MedMentions" $(FOCUS_CATS) | sha1sum))

show_focus_cats:
	@echo UMLS Categories
	@echo T116: "Amino Acid, Peptide, or Protein"
	@echo T126: Enzyme

	@echo DBPedia Categories
	@echo Agent
	@echo ChemicalSubstance

$(DATA_DIR):
	mkdir $(DATA_DIR)
$(DICT_DIR): $(DATA_DIR)
	mkdir -p $(DICT_DIR)

$(PSEUDO_DATA_DIR): $(DATA_DIR)
	echo $(PSEUDO_DATA_DIR)
	mkdir -p $(PSEUDO_DATA_DIR)

$(GOLD_DIR): $(DATA_DIR)
	mkdir -p $(GOLD_DIR)
$(GOLD_DIR)/MedMentions: $(GOLD_DIR)
	git clone https://github.com/chanzuckerberg/MedMentions
	for f in `find MedMentions/ | grep gz`; do gunzip $$f; done
	mv MedMentions $(GOLD_DIR)/MedMentions
$(GOLD_DATA): $(GOLD_DIR)/MedMentions
	poetry run python -m cli.preprocess.load_gold_ner $(subst $() ,_,$(FOCUS_CATS)) $(GOLD_DATA)


all: $(PSEUDO_NER_DATA_DIR) $(GOLD_DATA)

$(DICT_FILES): $(DICT_DIR)
	@echo make dict files $@
	poetry run python -m cli.preprocess.load_terms $@

$(RAW_CORPUS_DIR):
	mkdir -p $(RAW_CORPUS_DIR)
$(PUBMED): $(RAW_CORPUS_DIR)
	mkdir -p $(PUBMED)
	for f in `seq -w 1062`; do wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n$$f.xml.gz & done
	gunzip pubmed21n*.xml.gz
	mv pubmed21n*.xml $(PUBMED)
	for f in `ls $(PUBMED)/pubmed21n*.xml`; do poetry run python -m cli.preprocess.load_pubmed_txt $$f & done
	

$(RAW_CORPUS_OUT): $(SOURCE_TXT_DIR)
	@echo raw corpus num: $(RAW_CORPUS_NUM)
	@echo raw corpus out dir: $(RAW_CORPUS_OUT)
	poetry run python -m cli.preprocess.load_raw_corpus --raw-corpus-num $(RAW_CORPUS_NUM) --source-txt-dir $(SOURCE_TXT_DIR) --output-dir $(PSEUDO_NER_DATA_DIR)

$(PSEUDO_NER_DATA_DIR): $(DICT_FILES) $(PSEUDO_DATA_DIR) $(GOLD_DATA) $(RAW_CORPUS_OUT)
	@echo make pseudo ner data from $(DICT_FILES)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo output dir: $(PSEUDO_NER_DATA_DIR)
	poetry run python -m cli.preprocess.load_pseudo_ner --raw-corpus $(RAW_CORPUS_OUT) --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --duplicate-cats $(subst $() ,_,$(DUPLICATE_CATS)) --output_dir $(PSEUDO_NER_DATA_DIR)

# $(PSEUDO_SPAN_CLASSIF_DATA_DIR): $(PSEUDO_NER_DATA_DIR) $(PSEUDO_DATA_DIR)
# 	@echo make pseudo ner data translated into span classification from $(PSEUDO_NER_DATA_DIR).
# 	@echo focused categories: $(FOCUS_CATS)
# 	@echo duplicated categories: $(DUPLICATE_CATS)	
# 	@echo output dir: $(PSEUDO_SPAN_CLASSIF_DATA_DIR)
# 	poetry run python -m cli.preprocess.load_span_classif --ner-data $(PSEUDO_NER_DATA_DIR)