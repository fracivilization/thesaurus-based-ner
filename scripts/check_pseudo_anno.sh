get_make_cmd () {
    CMD="POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT} NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=${WITH_O} CHUNKER=${CHUNKER} make"
    echo ${CMD}
}
TMPFILE=$(mktemp)
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
WITH_O=False
CHUNKER="spacy_np"
MAKE=`get_make_cmd`
GOLD_NER_DATA_DIR=`eval ${MAKE} -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}'`
TERM2CAT=`eval ${MAKE} -n all | grep TERM2CAT | awk '{print $3}'`
# Exact Matching
poetry run python -m cli.train \
    ner_model=matcher \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} \
    +ner_model.term2cat=${TERM2CAT} \
    +testor.baseline_typer.term2cat=${TERM2CAT} 2>&1 | tee ${TMPFILE}
RUN_ID_ExactMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_ExactMatch" ${RUN_ID_ExactMatch}
CHUNKER="spacy_np"
# Ends with matching
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} \
    +ner_model.typer.term2cat=${TERM2CAT} \
    +testor.baseline_typer.term2cat=${TERM2CAT} 2>&1 | tee ${TMPFILE}
RUN_ID_EndsWithMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_EndsWithMatch" ${RUN_ID_EndsWithMatch}

# All Negatives (NP)
WITH_O=True
NEGATIVE_CATS=""
CHUNKER="spacy_np"
MAKE=`get_make_cmd`
GOLD_NER_DATA_DIR=`eval ${MAKE} -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}'`
TERM2CAT=`eval ${MAKE} -n all | grep TERM2CAT | awk '{print $3}'`
poetry run python -m cli.train \
    ner_model=matcher \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} \
    +ner_model.term2cat=${TERM2CAT} \
    +testor.baseline_typer.term2cat=${TERM2CAT} 2>&1 | tee ${TMPFILE}

# Thesaurus Negatives (UMLS + DBPedia)
CHUNKER="spacy_np"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200 GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=0.1
MAKE=`get_make_cmd`
# GOLD_NER_DATA_DIR=`eval ${MAKE} -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}'`
# TERM2CAT=`eval ${MAKE} -n all | grep TERM2CAT | awk '{print $3}'`
eval ${MAKE} train_pseudo_anno 2>&1 | tee ${TMPFILE}
# poetry run python -m cli.train \
#     ner_model=PseudoTwoStage \
#     ++dataset.name_or_path=${GOLD_NER_DATA_DIR} \
#     +ner_model.typer.term2cat=${TERM2CAT} \
#     +testor.baseline_typer.term2cat=${TERM2CAT} 2>&1 | tee ${TMPFILE}
RUN_ID_EndsWithMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus Negatives (UMLS + DBPedia)" ${RUN_ID_EndsWithMatch}

# Ends with matching into oracle term2cat
# poetry run python -m cli.train \
#     ner_model=PseudoTwoStage \
#     ner_model/typer/term2cat=oracle \
#     +ner_model.typer.term2cat.gold_dataset=${GOLD_NER_DATA_DIR} \
#     ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
# RUN_ID_OracleEndsWithMatch=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
# echo "RUN_ID_OracleEndsWithMatch" ${RUN_ID_OracleEndsWithMatch}