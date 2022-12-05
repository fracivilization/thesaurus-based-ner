
TERM2CAT := $(TERM2CAT_DIR)/$(firstword $(shell echo  "TERM2CAT" "FOCUS_CATS: $(FOCUS_CATS)" "NEGATIVE_CATS: $(NEGATIVE_CATS)" "POSITIVE_RATIO_THR_OF_NEGATIVE_CAT: ${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT}" | sha1sum)).pkl
UMLS_TERM2CATS := $(TERM2CATS_DIR)/$(firstword $(shell echo  "TERM2CATS" "FOCUS_CATS: $(UMLS_CATS)" | sha1sum)).pkl

PSEUDO_DATA_ARGS := $(TERM2CAT)
RUN_ARGS := $(O_SAMPLING_RATIO) $(FIRST_STAGE_CHUNKER)

DICT_FILES := $(addprefix $(DICT_DIR)/,$(APPEARED_CATS))
UMLS_DICT_FILES := $(addprefix $(DICT_DIR)/,$(UMLS_CATS))
# APPEARED_CATS を使って 出力先のフォルダを決める
RAW_CORPUS_OUT := $(RAW_CORPUS_DIR)/$(firstword $(shell echo $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo $(PSEUDO_DATA_ARGS) $(RAW_CORPUS_NUM) | sha1sum))
PSEUDO_MSC_NER_DATA_DIR := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MSC DATASET" $(PSEUDO_NER_DATA_DIR) $(WITH_O) $(FIRST_STAGE_CHUNKER) | sha1sum)) 
PSEUDO_OUT := outputs/$(firstword $(shell echo "PSEUDO_OUT" $(PSEUDO_DATA_ARGS) | sha1sum))

GOLD_DATA := $(GOLD_DIR)/$(firstword $(shell echo "MedMentions" $(FOCUS_CATS) | sha1sum))
GOLD_TRAIN_DATA := $(GOLD_DIR)/$(firstword $(shell echo "MedMentions" $(FOCUS_CATS) $(NEGATIVE_CATS) $(TRAIN_SNT_NUM) | sha1sum))
GOLD_TRAIN_MSC_DATA := $(GOLD_DIR)/$(firstword $(shell echo "GOLD MSC DATA" $(GOLD_TRAIN_DATA) $(MSC_ARGS) | sha1sum)) 
GOLD_MULTI_LABEL_NER_DATA := $(GOLD_DIR)/multi_label_ner
GOLD_MSMLC_BINARY_DATA := $(GOLD_DIR)/$(firstword $(shell echo "GOLD MSMLC BINARY DATA" $(GOLD_DATA) $(MSMLC_ARGS) | sha1sum)) 
GOLD_TRAIN_MSMLC_DATA := $(GOLD_DIR)/$(firstword $(shell echo "GOLD MSMLC DATA" $(GOLD_MULTI_LABEL_NER_DATA) $(MSMLC_ARGS) | sha1sum)) 
GOLD_TRAINED_MSMLC_BINARY_MODEL := $(MODEL_DIR)/$(firstword $(shell echo "GOLD TRAINED MSMLC MODEL" $(GOLD_TRAIN_MSMLC_DATA) | sha1sum)) 
GOLD_TRAINED_MSMLC_MODEL := $(MODEL_DIR)/$(firstword $(shell echo "GOLD TRAINED MSMLC MODEL" $(GOLD_TRAIN_MSMLC_DATA) $(MSMLC_ARGS) | sha1sum)) 


PSEUDO_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO_DATA_ON_GOLD" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum)) 
PSEUDO_MSC_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO MSC DATASET ON GOLD" $(PSEUDO_DATA_ON_GOLD) $(MSC_ARGS) | sha1sum)) 
PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO MULTI LABEL NER DATASET ON GOLD" $(UMLS_TERM2CATS) | sha1sum)) 
PSEUDO_MSMLC_DATA_ON_GOLD := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "PSEUDO MSMLC DATASET ON GOLD" $(PSEUDO_MULTI_LABEL_NER_DATA_ON_GOLD) $(MSMLC_ARGS) | sha1sum))  
PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL := $(MODEL_DIR)/$(firstword $(shell echo "PSEUDO ON GOLD TRAINED MSMLC MODEL" $(PSEUDO_MSMLC_DATA_ON_GOLD) $(MSMLC_ARGS) | sha1sum)) 
FP_REMOVED_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "FP_REMOVED_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
EROSION_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "EROSION_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))
MISGUIDANCE_PSEUDO_DATA := $(PSEUDO_DATA_DIR)/$(firstword $(shell echo "MISGUIDANCE_PSEUDO_DATA" $(PSEUDO_DATA_ARGS) $(GOLD_DATA) | sha1sum))

TRAIN_ON_GOLD_OUT := outputs/$(firstword $(shell echo "TRAIN_ON_GOLD_LOCK" $(GOLD_TRAIN_MSC_DATA) $(RUN_ARGS) | sha1sum))
TRAIN_OUT := outputs/$(firstword $(shell echo "TRAIN_LOCK" $(PSEUDO_MSC_DATA_ON_GOLD) $(RUN_ARGS) | sha1sum))
TRAIN_AND_EVAL_MSMLC_OUT := outputs/$(firstword $(shell echo "TRAIN_AND_EVAL_MSMLC_OUT" $(PSEUDO_MSMLC_DATA_ON_GOLD) $(MSMLC_ARGS) | sha1sum))
EVAL_FLATTEN_MARGINAL_MSMLC_OUT := outputs/$(firstword $(shell echo "TRAIN_AND_EVAL_MSMLC_OUT" $(PSEUDO_MSMLC_DATA_ON_GOLD) $(PSEUDO_ON_GOLD_TRAINED_MSMLC_MODEL) $(FOCUS_CATS) $(NEGATIVE_CATS) | sha1sum))
EVAL_FLATTEN_MARGINAL_MSMLC_ON_GOLD_OUT := outputs/$(firstword $(shell echo "TRAIN_AND_EVAL_MSMLC_OUT" $(GOLD_TRAIN_MSMLC_DATA) $(GOLD_TRAINED_MSMLC_MODEL) $(FOCUS_CATS) $(NEGATIVE_CATS) | sha1sum))