TMPFILE=$(mktemp)
WITH_NC=False
GOLD_NER_DATA_DIR=$(WITH_NC=${WITH_NC} make -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}')
# Exact Matching
poetry run python -m cli.train \
    ner_model=matcher \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_ExactMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_ExactMatch" ${RUN_ID_ExactMatch}

# Ends with matching
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_EndsWithMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_EndsWithMatch" ${RUN_ID_EndsWithMatch}

# Ends with matching into oracle term2cat
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ner_model/typer/term2cat=oracle \
    +ner_model.typer.term2cat.gold_dataset=${GOLD_NER_DATA_DIR} \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_OracleEndsWithMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_OracleEndsWithMatch" ${RUN_ID_OracleEndsWithMatch}