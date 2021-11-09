# All Negatives
PSEUDO_NER_DATA_DIR=$(make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
CHUNKER="enumerated"
RUN_OUT=$(python -m cli.train ner_model/chunker=${CHUNKER} ++dataset.name_or_path=${PSEUDO_NER_DATA_DIR})
echo $RUN_OUT
RUN_ID_WO_FP=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_WO_FP" ${RUN_ID_WO_FP}

python -m cli.compare_metrics --base-run-id ${RUN_ID_BASE} --focus-run-id ${RUN_ID_WO_FP}
# All Negatives (NP)
NO_NC=True
OUTPUT_O_AS_NC=True
NO_NC=${NO_NC} OUTPUT_O_AS_NC=${OUTPUT_O_AS_NC} make all -j$(nproc)
PSEUDO_DATA=$(NO_NC=${NO_NC} OUTPUT_O_AS_NC=${OUTPUT_O_AS_NC} make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
CHUNKER="spacy_np"
RUN_OUT=$(python -m cli.train ner_model/chunker=${CHUNKER} ++dataset.name_or_path=${FP_REMOVED_PSEUDO_DATA})
echo $RUN_OUT
RUN_ID_AllNegatives=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives (NP)" ${RUN_ID_AllNegatives (NP)}
# Thesaurus Negatives (UMLS)
PSEUDO_DATA=$(NO_NC=False OUTPUT_O_AS_NC=False make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
# Thesaurus Negatives (UMLS + DBPedia)