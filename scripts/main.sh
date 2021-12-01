TMPFILE=$(mktemp)

echo "PseudoAnno"
WITH_NC=True
GOLD_NER_DATA_DIR=$(WITH_NC=${WITH_NC} make -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}')
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_PseudoAnno=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}

get_enumerated_model_cmd () {
    CMD="\
        poetry run python -m cli.train \
        ++dataset.name_or_path=${RUN_DATASET} \
        ner_model/chunker=${CHUNKER} \
        ner_model.typer.msc_args.with_o=${WITH_O} \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_train_batch_size=8 \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True
    "
    echo ${CMD}
}
get_make_cmd () {
    CMD="WITH_NC=${WITH_NC} WITH_O=${WITH_O} CHUNKER=${CHUNKER} make"
    echo ${CMD}
}

echo "All Negatives"
# Get Dataset
WITH_NC=True
WITH_O=True
CHUNKER="enumerated"
MAKE=`get_make_cmd`
eval ${MAKE} all
RUN_DATASET=$(eval ${MAKE} -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
# Run
O_SAMPLING_RATIO=0.0001
CMD=`get_enumerated_model_cmd`
eval ${CMD} 2>&1 | tee ${TMPFILE}
RUN_ID_AllNegatives=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives" ${RUN_ID_AllNegatives}

echo "All Negatives (NP)"
WITH_NC=False
WITH_O=True
CHUNKER="spacy_np"
MAKE=`get_make_cmd`
RUN_DATASET=$(eval ${MAKE} -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
O_SAMPLING_RATIO=0.02
CMD=`get_enumerated_model_cmd`
eval ${CMD} 2>&1 | tee ${TMPFILE}
RUN_ID_AllNegatives_NP=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives (NP)" ${RUN_ID_AllNegatives_NP}

echo "Thesaurus Negatives (UMLS)"
WITH_NC=True
WITH_O=False
CHUNKER="spacy_np"
MAKE=`get_make_cmd`
RUN_DATASET=$(eval ${MAKE} -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
O_SAMPLING_RATIO=0.00
CMD=`get_enumerated_model_cmd`
eval ${CMD} 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


echo "Thesaurus Negatives (UMLS) + All Negatives (NP)"
WITH_NC=True
WITH_O=True
CHUNKER="spacy_np"
MAKE=`get_make_cmd`
RUN_DATASET=$(eval ${MAKE} -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
O_SAMPLING_RATIO=0.02
CMD=`get_enumerated_model_cmd`
eval ${CMD} 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


# Thesaurus Negatives (UMLS + DBPedia)