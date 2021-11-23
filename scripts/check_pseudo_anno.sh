TMPFILE=$(mktemp)
NO_NC=True
NO_NC=${NO_NC} make all -j$(nproc)
GOLD_NER_DATA_DIR=$(NO_NC=${NO_NC} make -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}')
# Exact Matching
poetry run python -m cli.train \
    ner_model=matcher \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_PseudoAnno=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}

# Ends with matching
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_PseudoAnno=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}

# Ends with matching into oracle term2cat
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ner_model/chunker=spacy_np \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_PseudoAnno=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}