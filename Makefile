# pseudo dataset related args
DBPEDIA_CATS = GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour
UMLS_CATS = T002 T004 T194 T075 T200 T081 T080 T079 T171 T102 T099 T100 T101 T054 T055 T056 T064 T065 T066 T068 T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
FOCUS_CATS ?= T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
NEGATIVE_CATS ?= T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200 $(DBPEDIA_CATS)
# WITH_NC ?= True
WITH_O ?= True
CHUNKER ?= spacy_np
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT ?= 1.0
O_SAMPLING_RATIO ?= 1.0
MSC_O_SAMPLING_RATIO ?= 1.0
TRAIN_SNT_NUM ?= 9223372036854775807

MSC_ARGS := "WITH_O: $(WITH_O) CHUNKER: $(CHUNKER) MSC_O_SAMPLING_RATIO: $(MSC_O_SAMPLING_RATIO)"

DATA_DIR := data
TERM2CAT_DIR := $(DATA_DIR)/term2cat

TERM2CAT := $(TERM2CAT_DIR)/$(firstword $(shell echo  "TERM2CAT" "FOCUS_CATS: $(FOCUS_CATS)" "NEGATIVE_CATS: $(NEGATIVE_CATS)" "POSITIVE_RATIO_THR_OF_NEGATIVE_CAT: ${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT}" | sha1sum)).pkl

PSEUDO_DATA_ARGS := $(TERM2CAT)
RUN_ARGS := $(O_SAMPLING_RATIO) $(CHUNKER)

APPEARED_CATS := $(FOCUS_CATS) $(NEGATIVE_CATS)
DICT_DIR := $(DATA_DIR)/dict
DICT_FILES := $(addprefix $(DICT_DIR)/,$(APPEARED_CATS))
UMLS_DIR := $(DATA_DIR)/2021AA-full
DBPEDIA_DIR := $(DATA_DIR)/DBPedia
PubChem_DIR := $(DATA_DIR)/PubChem
RAW_SENTENCE_NUM := 50000
# APPEARED_CATS を使って 出力先のフォルダを決める
RAW_CORPUS_DIR := $(DATA_DIR)/raw
PUBMED := $(RAW_CORPUS_DIR)/pubmed
SOURCE_TXT_DIR := $(PUBMED)
RAW_CORPUS_OUT := $(RAW_CORPUS_DIR)/$(firstword $(shell echo $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_DATA_DIR := $(DATA_DIR)/pseudo
PSEUDO_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(PSEUDO_DATA_ARGS) $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_MSC_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MSC DATASET" $(PSEUDO_NER_DATA_DIR) $(WITH_O) $(CHUNKER) | sha1sum)) 

GOLD_DIR := $(DATA_DIR)/gold
GOLD_DATA := $(GOLD_DIR)/$(firstword $(shell echo "MedMentions" $(FOCUS_CATS) $(TRAIN_SNT_NUM) | sha1sum))
GOLD_MSC_DATA := $(GOLD_DIR)/$(firstword $(shell echo "GOLD MSC DATA" $(GOLD_DATA) $(MSC_ARGS) | sha1sum)) 

PSEUDO_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_pseudo_ner \
		++ner_model.typer.term2cat=$(TERM2CAT) \
        +gold_corpus=$(GOLD_DATA)
MSC_DATA_BASE_CMD := poetry run python -m cli.preprocess.load_msc_dataset chunker=$(CHUNKER) ++with_o=$(WITH_O) ++o_sampling_ratio=$(MSC_O_SAMPLING_RATIO)


PSEUDO_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO_DATA_ON_GOLD" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum)) 
PSEUDO_MSC_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MSC DATASET ON GOLD" $(PSEUDO_DATA_ON_GOLD) $(MSC_ARGS) | sha1sum)) 
FP_REMOVED_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "FP_REMOVED_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
EROSION_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "EROSION_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
MISGUIDANCE_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MISGUIDANCE_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))

TRAIN_ON_GOLD_OUT := outputs/$(firstword $(shell echo "TRAIN_ON_GOLD_LOCK" $(GOLD_MSC_DATA) $(RUN_ARGS) | sha1sum))
TRAIN_OUT := outputs/$(firstword $(shell echo "TRAIN_LOCK" $(PSEUDO_MSC_DATA_ON_GOLD) $(RUN_ARGS) | sha1sum))

TRAIN_BASE_CMD := poetry run python -m cli.train \
		++dataset.name_or_path=$(GOLD_DATA) \
		ner_model/chunker=$(CHUNKER) \
		ner_model.typer.model_args.o_sampling_ratio=$(O_SAMPLING_RATIO) \
		ner_model.typer.train_args.per_device_train_batch_size=8 \
		ner_model.typer.train_args.per_device_eval_batch_size=32 \
		ner_model.typer.train_args.do_train=True \
		ner_model.typer.train_args.overwrite_output_dir=True \
		testor.baseline_typer.term2cat=$(TERM2CAT)
test: 
	@echo MSC_O_SAMPLING_RATIO: $(MSC_O_SAMPLING_RATIO)
	@echo MSC_ARGS: $(MSC_ARGS)
term2cat: $(TERM2CAT)
	@echo TERM2CAT: $(TERM2CAT)


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
$(PubChem_DIR): $(DATA_DIR)
	mkdir -p $(PubChem_DIR)
	wget https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/CURRENT-Full/XML/
	FILES=`cat index.html  | grep xml.gz | grep -v md5| sed -e 's/<[^>]*>//g' | awk '{print $$1}'`
	for f in `cat index.html  | grep xml.gz | grep -v md5| sed -e 's/<[^>]*>//g' | awk '{print $1}'`; do 
		wget "https://ftp.ncbi.nlm.nih.gov/pubchem/Substance/CURRENT-Full/XML/${f}"
	done
	gunzip *.gz
	mv *.xml $(PubChem_DIR)/

$(TERM2CAT_DIR): $(DATA_DIR)
	mkdir -p $(TERM2CAT_DIR)
$(TERM2CAT): $(TERM2CAT_DIR) $(DICT_FILES)
	@echo TERM2CAT: $(TERM2CAT)
	poetry run python -m cli.preprocess.load_term2cat \
		output=$(TERM2CAT) \
		focus_cats=$(subst $() ,_,$(FOCUS_CATS)) \
		negative_cats=$(subst $() ,_,$(NEGATIVE_CATS)) \
		++positive_ratio_thr_of_negative_cat=${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT}

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
	@poetry run python -m cli.preprocess.load_gold_ner --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_DIR)/MedMentions/st21pv/data --train-snt-num $(TRAIN_SNT_NUM)
	poetry run python -m cli.preprocess.load_gold_ner --focus-cats $(subst $() ,_,$(FOCUS_CATS)) --output $(GOLD_DATA) --input-dir $(GOLD_DIR)/MedMentions/st21pv/data --train-snt-num $(TRAIN_SNT_NUM)
$(GOLD_MSC_DATA): $(GOLD_DATA)
	@echo GOLD_MSC_DATA_ON_GOLD: $(GOLD_MSC_DATA)
	$(MSC_DATA_BASE_CMD) \
		+ner_dataset=$(GOLD_DATA) \
		+output_dir=$(GOLD_MSC_DATA)


all: ${DICT_FILES} $(PSEUDO_NER_DATA_DIR) $(PSEUDO_MSC_NER_DATA_DIR) $(GOLD_DATA) $(GOLD_MSC_DATA) $(PSEUDO_DATA_ON_GOLD) $(PSEUDO_MSC_DATA_ON_GOLD) $(FP_REMOVED_PSEUDO_DATA)

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

$(PSEUDO_NER_DATA_DIR): $(DICT_FILES) $(PSEUDO_DATA_DIR) $(GOLD_DATA) $(RAW_CORPUS_OUT) $(TERM2CAT)
	@echo make pseudo ner data from $(DICT_FILES)
	@echo focused categories: $(FOCUS_CATS)
	@echo negative categories: $(NEGATIVE_CATS)
	@echo PSEUDO_NER_DATA_DIR: $(PSEUDO_NER_DATA_DIR)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(RAW_CORPUS_OUT) \
		+output_dir=$(PSEUDO_NER_DATA_DIR)
$(PSEUDO_MSC_NER_DATA_DIR): $(PSEUDO_NER_DATA_DIR)
	@echo PSEUDO_MSC_NER_DATA_DIR: $(PSEUDO_MSC_NER_DATA_DIR)
	$(MSC_DATA_BASE_CMD) \
		+ner_dataset=$(PSEUDO_DATA_ON_GOLD) \
		+output_dir=$(PSEUDO_MSC_NER_DATA_DIR)

        
$(FP_REMOVED_PSEUDO_DATA): $(DICT_FILES) $(GOLD_DATA) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR) $(TERM2CAT)
	@echo make pseudo data whose FP is removed according to Gold dataset
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
	@echo negative categories: $(NEGATIVE_CATS)
	@echo FP_REMOVED_PSEUDO_DATA: $(FP_REMOVED_PSEUDO_DATA)
	$(PSEUDO_DATA_BASE_CMD) \
		+raw_corpus=$(GOLD_DATA) \
		+output_dir=$(FP_REMOVED_PSEUDO_DATA) \
		++remove_fp_instance=True

$(PSEUDO_DATA_ON_GOLD): $(GOLD_DATA) $(DICT_FILES) $(PSEUDO_DATA_DIR) $(PSEUDO_NER_DATA_DIR) $(TERM2CAT)
	@echo make pseudo data on Gold dataset for comparison
	@echo make from Gold: $(GOLD_DATA)
	@echo focused categories: $(FOCUS_CATS)
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

$(TRAIN_OUT): $(PSEUDO_MSC_DATA_ON_GOLD) $(TERM2CAT)
	$(TRAIN_BASE_CMD) \
		ner_model.typer.msc_datasets=$(PSEUDO_MSC_DATA_ON_GOLD) 2>&1 | tee $(TRAIN_OUT)
train: $(TRAIN_OUT)
	@echo TRAIN_OUT: $(TRAIN_OUT)

$(TRAIN_ON_GOLD_OUT): $(GOLD_MSC_DATA) $(TERM2CAT)
	$(TRAIN_BASE_CMD) \
		ner_model.typer.msc_datasets=$(GOLD_MSC_DATA) 2>&1 | tee $(TRAIN_ON_GOLD_OUT)
train_on_gold: $(TRAIN_ON_GOLD_OUT) 
	@echo TRAIN_ON_GOLD_OUT: $(TRAIN_ON_GOLD_OUT)

train_pseudo_anno: $(GOLD_DATA) $(TERM2CAT)
	poetry run python -m cli.train \
		ner_model=PseudoTwoStage \
		++dataset.name_or_path=$(GOLD_DATA) \
		+ner_model.typer.term2cat=$(TERM2CAT) \
		+testor.baseline_typer.term2cat=$(TERM2CAT)