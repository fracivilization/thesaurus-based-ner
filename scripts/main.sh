# PseudoAnno
NO_NC=True
NO_NC=${NO_NC} make all -j$(nproc)
GOLD_NER_DATA_DIR=$(NO_NC=${NO_NC} make -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}')
RUN_OUT=$(
    poetry run python -m cli.train \
        ner_model=PseudoTwoStage \
        ++dataset.name_or_path=${GOLD_NER_DATA_DIR} \
)
echo $RUN_OUT
RUN_ID_PseudoAnno=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}

# All Negatives
NO_NC=True
O_SAMPLING_RATIO=0.0002
NO_NC=${NO_NC} make all -j$(nproc)
PSEUDO_NER_DATA_DIR=$(NO_NC=${NO_NC} make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
CHUNKER="enumerated"
RUN_OUT=$(
    poetry run python -m cli.train \
        ner_model/chunker=${CHUNKER} \
        ++dataset.name_or_path=${PSEUDO_NER_DATA_DIR} \
        ner_model.typer.msc_args.with_enumerated_o_label=True \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.per_device_train_batch_size=16 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
)
echo $RUN_OUT
RUN_ID_AllNegatives=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives" ${RUN_ID_AllNegatives}

# All Negatives (NP)
NO_NC=True
O_SAMPLING_RATIO=0.01
PSEUDO_DATA=$(NO_NC=${NO_NC}  make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
NO_NC=${NO_NC} make all -j$(nproc)
CHUNKER="spacy_np"

RUN_OUT=$(
    poetry run python -m cli.train \
        ner_model/chunker=${CHUNKER} \
        ++dataset.name_or_path=${PSEUDO_DATA} \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
        ner_model.typer.train_args.per_device_train_batch_size=16 \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
)
echo $RUN_OUT
RUN_ID_AllNegatives_NP=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives (NP)" ${RUN_ID_AllNegatives_NP}

# Thesaurus Negatives (UMLS)
NO_NC=False
NO_NC=${NO_NC} make all -j$(nproc)
PSEUDO_DATA=$(NO_NC=${NO_NC}  make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
CHUNKER="spacy_np"
RUN_OUT=$(
    poetry run python -m cli.train \
        ner_model/chunker=${CHUNKER} \
        ++dataset.name_or_path=${PSEUDO_DATA} \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
        ner_model.typer.train_args.per_device_train_batch_size=8 \
        ner_model.typer.train_args.per_device_eval_batch_size=16 \
)
echo $RUN_OUT
RUN_ID_Thesaurus_Negatives_UMLS=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}
# Thesaurus Negatives (UMLS + DBPedia)