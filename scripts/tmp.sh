test () {
    echo "sdfs sdfsdf" | awk '{print $1}'
    OPTIONS="sdf sdfsd sdffsd"
    echo "sdfds ${OPTIONS}"
}
# test

get_cmd () {
    CMD="\
        poetry run python -m cli.train \
        ++dataset.name_or_path=${PSEUDO_DATA_ON_GOLD}) \
        ner_model/chunker=${CHUNKER} \
        ner_model.typer.msc_args.with_enumerated_o_label=${WITH_ENUMERATED_O} \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_train_batch_size=8 \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True | 2>&1 | tee ${TMPFILE}
    "
    echo ${CMD}
}

CMD=`get_cmd`
echo $CMD