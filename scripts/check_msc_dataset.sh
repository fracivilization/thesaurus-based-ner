BASE_OPTS="WITH_NC=True WITH_O=False CHUNKER=spacy_np"
BASE_MSC_DATA=`eval ${BASE_OPTS} make -n all | grep PSEUDO_MSC_DATA_ON_GOLD | awk '{print $3}'`
FOCUS_OPTS="WITH_NC=True WITH_O=True CHUNKER=spacy_np"
FOCUS_MSC_DATA=`eval ${FOCUS_OPTS} make -n all | grep PSEUDO_MSC_DATA_ON_GOLD | awk '{print $3}'`

OUTPUT_DIR=outputs/error_analysis/check_msc_dataset
echo ${BASE_MSC_DATA} ${FOCUS_MSC_DATA} ${OUTPUT_DIR}
poetry run python -m cli.compare_msc_dataset \
    --base-msc-datasets \
    ${BASE_MSC_DATA} \
    --focus-msc-datasets \
    ${FOCUS_MSC_DATA} \
    --output-dir \
    ${OUTPUT_DIR} > ${OUTPUT_DIR}/cout


