# Base Score
## Base Datasetを作成
NO_NC=True
OUTPUT_O_AS_NC=False
O_SAMPLING_RATIO=0.0002
NO_NC=${NO_NC} OUTPUT_O_AS_NC=${OUTPUT_O_AS_NC} make all -j$(nproc)
PSEUDO_NER_DATA_DIR=$(NO_NC=${NO_NC} OUTPUT_O_AS_NC=${OUTPUT_O_AS_NC} make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
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
## その上で走らせる & RunIDを記録

# Erosion Rate

EROSION_PSEUDO_DATA=$(NO_NC=${NO_NC} OUTPUT_O_AS_NC=${OUTPUT_O_AS_NC} make -n all | grep EROSION_PSEUDO_DATA | awk '{print $3}')
## その上で走らせる & RunIDを記録
RUN_OUT=$(
    poetry run python -m cli.train \
        ner_model/chunker=${CHUNKER} \
        ++dataset.name_or_path=${EROSION_PSEUDO_DATA} \
        ner_model.typer.msc_args.with_enumerated_o_label=True \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.per_device_train_batch_size=16 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
)
echo $RUN_OUT
RUN_ID_AllNegatives=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
## Base Scoreと比較

# Misguidance Rate
## Misguidance Datasetを作成
## その上で走らせる & RunIDを記録
## Base Scoreと比較