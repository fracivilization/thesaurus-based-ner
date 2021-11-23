# pseudo dataset related args
DBPEDIA_CATS = GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour
UMLS_CATS = T002 T004 T194 T075 T200 T081 T080 T079 T171 T102 T099 T100 T101 T054 T055 T056 T064 T065 T066 T068 T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
FOCUS_CATS := T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
DUPLICATE_CATS := $(DBPEDIA_CATS)
NO_NC ?= False

PSEUDO_DATA_ARGS := $(FOCUS_CATS) $(DUPLICATE_CATS) $(NO_NC) 

REMOVE_CATS := $(filter-out $(FOCUS_CATS), $(filter-out $(DUPLICATE_CATS), $(DBPEDIA_CATS) $(UMLS_CATS)))
APPEARED_CATS := $(FOCUS_CATS) $(REMOVE_CATS)
DATA_DIR := data
DICT_DIR := $(DATA_DIR)/dict
DICT_FILES := $(addprefix $(DICT_DIR)/,$(APPEARED_CATS))
UMLS_DIR := $(DATA_DIR)/2021AA-full
DBPEDIA_DIR := $(DATA_DIR)/DBPedia
RAW_SENTENCE_NUM := 50000
# APPEARED_CATS を使って 出力先のフォルダを決める
RAW_CORPUS_DIR := $(DATA_DIR)/raw
PUBMED := $(RAW_CORPUS_DIR)/pubmed
SOURCE_TXT_DIR := $(PUBMED)
RAW_CORPUS_OUT := $(RAW_CORPUS_DIR)/$(firstword $(shell echo $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_DATA_DIR := $(DATA_DIR)/pseudo
PSEUDO_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(PSEUDO_DATA_ARGS) $(RAW_CORPUS_NUM) | sha1sum))

GOLD_DIR := $(DATA_DIR)/gold
GOLD_DATA := $(GOLD_DIR)/$(firstword $(shell echo "MedMentions" $(FOCUS_CATS) | sha1sum))

PSEUDO_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO_DATA_ON_GOLD" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum)) 
FP_REMOVED_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "FP_REMOVED_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
EROSION_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "EROSION_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
MISGUIDANCE_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MISGUIDANCE_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))

PSEUDO_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_pseudo_ner \
		++ner_model.typer.term2cat.focus_cats=$(subst $() ,_,$(FOCUS_CATS)) \
		++ner_model.typer.term2cat.duplicate_cats=$(subst $() ,_,$(DUPLICATE_CATS)) \
		++ner_model.typer.term2cat.no_nc=$(NO_NC) \
        +gold_corpus=$(GOLD_DATA)

test:
	$(PSEUDO_DATA_BASE_CMD)
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
$(UMLS_DIR): $(DATA_DIR)
	@echo "Please Download UMLS2021AA full from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html"
	@echo "You need UMLS Account, so please acces by web browser and mv the file into $(UMLS_DIR)"
	# unzip $(UMLS_DIR)/mmsys.zip
	@echo "Plaese refer to README.md"
$(DBPEDIA_DIR): $(DATA_DIR)
	mkdir -p $(DBPEDIA_DIR)
	# wget https://databus.dbpedia.org/ontologies/dbpedia.org/ontology--DEV/2021.07.09-070001/ontology--DEV_type=parsed_sorted.nt # DBPedia Ontlogy
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
	# bunzip2 *.bz2
	# mv *.ttl $(DBPEDIA_DIR)
	# mv *.nt $(DBPEDIA_DIR)

$(PSEUDO_DATA_DIR): $(DATA_DIR)
	echo $(PSEUDO_DATA_DIR)
	mkdir -p $(PSEUDO_DATA_DIR)

$(GOLD_DIR): $(DATA_DIR)
	mkdir -p $(GOLD_DIR)
$(GOLD_DIR)/MedMentions: $(GOLD_DIR)
	# git clone https://github.com/chanzuckerberg/MedMentions
	# for f in `find MedMentions/ | grep gz`; do gunzip $$f; done
	# mv MedMentions $(GOLD_DIR)/MedMentions
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_trng.txt data/gold/MedMentions/st21pv/data/
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_dev.txt data/gold/MedMentions/st21pv/data/
	# cp data/gold/MedMentions/full/data/corpus_pubtator_pmids_test.txt data/gold/MedMentions/st21pv/data/
$(GOLD_DATA): $(GOLD_DIR)/MedMentions
	@echo "Gold Data"
	@echo GOLD_NER_DATA_DIR: $(GOLD_DATA)
	@poetry run python -m cli.preprocess.load_gold_ner --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_DIR)/MedMentions/st21pv/data
	poetry run python -m cli.preprocess.load_gold_ner --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_DIR)/MedMentions/st21pv/data


all: $(PSEUDO_NER_DATA_DIR) $(GOLD_DATA) $(PSEUDO_DATA_ON_GOLD) $(FP_REMOVED_PSEUDO_DATA) $(EROSION_PSEUDO_DATA) $(MISGUIDANCE_PSEUDO_DATA)
	@echo $(APPEARED_CATS)

$(DICT_FILES): $(DICT_DIR) $(UMLS_DIR) $(DBPEDIA_DIR)
	@echo make dict files $@
	poetry run python -m cli.preprocess.load_terms --category $(notdir $@) --output $@

$(RAW_CORPUS_DIR):
	mkdir -p $(RAW_CORPUS_DIR)
$(PUBMED): $(RAW_CORPUS_DIR)
	# mkdir -p $(PUBMED)
	# for f in `seq -w 1062`; do wget https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed21n$$f.xml.gz ; gunzip pubmed21n$$f.xml.gz & done
	# mv pubmed21n*.xml $(PUBMED)
	# for f in `ls $(PUBMED)/pubmed21n*.xml`; do poetry run python -m cli.preprocess.load_pubmed_txt $$f & done
	

$(RAW_CORPUS_OUT): $(SOURCE_TXT_DIR)
	@echo raw sentence num: $(RAW_SENTENCE_NUM)
	@echo raw corpus out dir: $(RAW_CORPUS_OUT)
	poetry run python -m cli.preprocess.load_raw_corpus --raw-sentence-num $(RAW_SENTENCE_NUM) --source-txt-dir $(SOURCE_TXT_DIR) --output-dir $(RAW_CORPUS_OUT)

$(PSEUDO_NER_DATA_DIR): $(DICT_FILES) $(PSEUDO_DATA_DIR) $(GOLD_DATA) $(RAW_CORPUS_OUT)
	@echo make pseudo ner data from $(DICT_FILES)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo PSEUDO_NER_DATA_DIR: $(PSEUDO_NER_DATA_DIR)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(RAW_CORPUS_OUT) \
		+output_dir=$(PSEUDO_NER_DATA_DIR)
        
$(FP_REMOVED_PSEUDO_DATA): $(DICT_FILES) $(GOLD_DATA) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR)
	@echo make pseudo data whose FP is removed according to Gold dataset
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo FP_REMOVED_PSEUDO_DATA: $(FP_REMOVED_PSEUDO_DATA)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(FP_REMOVED_PSEUDO_DATA) \
		++remove_fp_instance=True

$(PSEUDO_DATA_ON_GOLD): $(GOLD_DATA) $(DICT_FILES) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR)
	@echo make pseudo data on Gold dataset for comparison
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo PSEUDO_DATA_ON_GOLD: $(PSEUDO_DATA_ON_GOLD)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(PSEUDO_DATA_ON_GOLD) \

$(EROSION_PSEUDO_DATA):  $(GOLD_DATA) $(DICT_FILES) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR)
	@echo make pseudo data for erosion experiment
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo pseudo_data_on_gold: $(PSEUDO_DATA_ON_GOLD)
	@echo EROSION_PSEUDO_DATA: $(EROSION_PSEUDO_DATA)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(EROSION_PSEUDO_DATA) \
		++add_erosion_fn=True


$(MISGUIDANCE_PSEUDO_DATA):  $(GOLD_DATA) $(DICT_FILES) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR)
	@echo make pseudo data for erosion experiment
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
	@echo duplicated categories: $(DUPLICATE_CATS)
	@echo pseudo_data_on_gold: $(PSEUDO_DATA_ON_GOLD)
	@echo MISGUIDANCE_PSEUDO_DATA: $(MISGUIDANCE_PSEUDO_DATA)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(MISGUIDANCE_PSEUDO_DATA) \
		++remove_misguidance_fn=True
# $(PSEUDO_SPAN_CLASSIF_DATA_DIR): $(PSEUDO_NER_DATA_DIR) $(PSEUDO_DATA_DIR)
# 	@echo make pseudo ner data translated into span classification from $(PSEUDO_NER_DATA_DIR).
# 	@echo focused categories: $(FOCUS_CATS)
# 	@echo duplicated categories: $(DUPLICATE_CATS)	
# 	@echo output dir: $(PSEUDO_SPAN_CLASSIF_DATA_DIR)
# 	poetry run python -m cli.preprocess.load_span_classif --ner-data $(PSEUDO_NER_DATA_DIR)