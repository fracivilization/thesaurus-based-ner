# Constants
# pseudo dataset related args
DBPEDIA_CATS = GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour
UMLS_CATS = T000 T116 T020 T052 T100 T087 T011 T190 T008 T017 T195 T194 T123 T007 T031 T022 T053 T038 T012 T029 T091 T122 T023 T030 T026 T043 T025 T019 T103 T120 T104 T185 T201 T200 T077 T049 T088 T060 T056 T203 T047 T065 T069 T196 T050 T018 T071 T126 T204 T051 T099 T021 T013 T033 T004 T168 T169 T045 T083 T028 T064 T102 T096 T068 T093 T058 T131 T125 T016 T078 T129 T055 T197 T037 T170 T130 T171 T059 T034 T015 T063 T066 T074 T041 T073 T048 T044 T085 T191 T114 T070 T086 T057 T090 T109 T032 T040 T001 T092 T042 T046 T072 T067 T039 T121 T002 T101 T098 T097 T094 T080 T081 T192 T014 T062 T075 T089 T167 T095 T054 T184 T082 T024 T079 T061 T005 T127 T010
RAW_SENTENCE_NUM := 50000

APPEARED_CATS := $(FOCUS_CATS) $(NEGATIVE_CATS)

# Dirs
DATA_DIR := data
TERM2CAT_DIR := $(DATA_DIR)/term2cat
TERM2CATS_DIR := $(DATA_DIR)/term2cats
DICT_DIR := $(DATA_DIR)/dict
UMLS_DIR := $(DATA_DIR)/2021AA
DBPEDIA_DIR := $(DATA_DIR)/DBPedia
RAW_CORPUS_DIR := $(DATA_DIR)/raw
BUFFER_DIR := $(DATA_DIR)/buffer
PUBMED := $(RAW_CORPUS_DIR)/pubmed
SOURCE_TXT_DIR := $(PUBMED)
PSEUDO_DATA_DIR := $(DATA_DIR)/pseudo
MODEL_DIR := $(DATA_DIR)/model
GOLD_DIR := $(DATA_DIR)/gold
MED_MENTIONS_DIR := $(GOLD_DIR)/MedMentions
CONLL2003_DIR := $(GOLD_DIR)/CoNLL2003