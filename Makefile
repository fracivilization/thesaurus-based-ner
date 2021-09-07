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
PSEUDO_DATA_DIR := $(DATA_DIR)/pseudo
PSEUDO_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(APPEARED_CATS) $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_SPAN_CLASSIF_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(PSEUDO_NER_DATA_DIR) "SPAN" | sha1sum))

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
	mkdir $(GOLD_DIR)
$(GOLD_DIR)/MedMentions: $(GOLD_DIR)
	git clone https://github.com/chanzuckerberg/MedMentions
	mv MedMentions $(GOLD_DIR)/MedMentions
$(GOLD_DATA): $(GOLD_DIR)/MedMentions
	poetry run python -m cli.preprocess.load_gold_ner $(subst $() ,_,$(FOCUS_CATS))

all: $(PSEUDO_NER_DATA_DIR) $(PSEUDO_SPAN_CLASSIF_DATA_DIR) $(GOLD_DATA)

$(DICT_FILES): $(DICT_DIR)
	@echo make dict files $@
	poetry run python -m cli.preprocess.load_terms $@

$(PSEUDO_NER_DATA_DIR): $(DICT_FILES) $(PSEUDO_DATA_DIR) $(GOLD_DATA)
	@echo make pseudo ner data from $(DICT_FILES)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo raw corpus num: $(RAW_CORPUS_NUM)
	@echo output dir: $(PSEUDO_NER_DATA_DIR)
	poetry run python -m cli.preprocess.load_pseudo_ner --raw-corpus-num $(RAW_CORPUS_NUM) --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --duplicate-cats $(subst $() ,_,$(DUPLICATE_CATS))

# $(PSEUDO_SPAN_CLASSIF_DATA_DIR): $(PSEUDO_NER_DATA_DIR) $(PSEUDO_DATA_DIR)
# 	@echo make pseudo ner data translated into span classification from $(PSEUDO_NER_DATA_DIR).
# 	@echo focused categories: $(FOCUS_CATS)
# 	@echo duplicated categories: $(DUPLICATE_CATS)	
# 	@echo output dir: $(PSEUDO_SPAN_CLASSIF_DATA_DIR)
# 	poetry run python -m cli.preprocess.load_span_classif --ner-data $(PSEUDO_NER_DATA_DIR)