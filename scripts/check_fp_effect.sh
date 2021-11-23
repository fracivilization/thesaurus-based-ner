TMPFILE=$(mktemp)

get_cmd () {
    CMD="\
        poetry run python -m cli.train \
        ++dataset.name_or_path=${RUN_DATASET} \
        ner_model/chunker=${CHUNKER} \
        ner_model.typer.msc_args.with_enumerated_o_label=${WITH_ENUMERATED_O} \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_train_batch_size=8 \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
    "
    echo ${CMD}
}

check_fp_effect () {
    NO_NC=${NO_NC} make all -j$(nproc)
    RUN_DATASET=$(NO_NC=${NO_NC} make -n all | grep PSEUDO_DATA_ON_GOLD | awk '{print $3}')
    CMD=`get_cmd`
    echo ${CMD}
    eval ${CMD} 2>&1 | tee ${TMPFILE}
    RUN_ID_BASE=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
    echo "RUN_ID_BASE" ${RUN_ID_BASE}

    RUN_DATASET=$(NO_NC=${NO_NC} make -n all | grep FP_REMOVED_PSEUDO_DATA | awk '{print $3}')
    CMD=`get_cmd`
    echo ${CMD}
    eval ${CMD} 2>&1 | tee ${TMPFILE}
    RUN_ID_WO_FP=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
    echo "RUN_ID_WO_FP" ${RUN_ID_WO_FP}

    poetry run python -m cli.compare_metrics --base-run-id ${RUN_ID_BASE} --focus-run-id ${RUN_ID_WO_FP}
}



echo "All Negatives"
NO_NC=True
O_SAMPLING_RATIO=0.0001
WITH_ENUMERATED_O=True
CHUNKER="enumerated"
check_fp_effect

echo "All Negatives (NP)"
NO_NC=True
O_SAMPLING_RATIO=0.02
WITH_ENUMERATED_O=True
CHUNKER="spacy_np"
check_fp_effect

echo "Thesaurus Negatives (UMLS)"
NO_NC=False
WITH_ENUMERATED_O=False
O_SAMPLING_RATIO=0.0 # This variable isn't needed but, I added for get_cmd function
CHUNKER="spacy_np"
check_fp_effect
# Thesaurus Negatives (UMLS + DBPedia)