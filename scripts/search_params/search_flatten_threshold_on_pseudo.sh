#!/bin/bash
# BASE="poetry run python -m cli.train ner_model=flatten_ner ner_model.focus_cats=T005_T007_T017_T022_T031_T033_T037_T038_T058_T062_T074_T082_T091_T092_T097_T098_T103_T168_T170_T201_T204 \
# ner_model/multi_label_ner_model=two_stage +ner_model/multi_label_ner_model/chunker=enumerated \
# +ner_model/multi_label_ner_model/multi_label_typer=enumerated \
# ++ner_model.multi_label_ner_model.multi_label_typer.model_args.saved_param_path=outputs/2022-02-11/18-00-49/checkpoint-2000 \
# ++ner_model.multi_label_ner_model.multi_label_typer.train_datasets=data/gold/59eb531b4a37f968e1af817a39c850629a694987 \
# ++ner_model.multi_label_ner_model.multi_label_typer.train_args.do_train=False ++ner_model.multi_label_ner_model.multi_label_typer.prediction_threshold=0.7 testor.baseline_typer.term2cat=data/term2cat/8a23fbd2bc56b5182ab677063f52af0497d1d5c6.pkl"


dir=`dirname $0`
thresholds=(0.6 0.7 0.8 0.9 0.95 0.96 0.97 0.98)
# negative_ratios=(32.0 64.0 128.0)
for threshold in ${thresholds[@]}; do
    echo threshold: ${threshold}
    poetry run python -m cli.train \
        ner_model=flatten_ner \
        ner_model.focus_cats=T005_T007_T017_T022_T031_T033_T037_T038_T058_T062_T074_T082_T091_T092_T097_T098_T103_T168_T170_T201_T204 \
        ner_model/multi_label_ner_model=two_stage \
        +ner_model/multi_label_ner_model/chunker=enumerated \
        +ner_model/multi_label_ner_model/multi_label_typer=enumerated \
        testor.baseline_typer.term2cat=data/term2cat/8a23fbd2bc56b5182ab677063f52af0497d1d5c6.pkl \
        ++ner_model.multi_label_ner_model.multi_label_typer.model_args.saved_param_path=outputs/2022-02-16/12-30-34/checkpoint-2000 \
        ++ner_model.multi_label_ner_model.multi_label_typer.train_args.do_train=False \
        ++ner_model.multi_label_ner_model.multi_label_typer.train_datasets=data/pseudo/fe237f63d08d4a49a18f9e3f4a3316e16123d288 \
        ++ner_model.multi_label_ner_model.multi_label_typer.prediction_threshold=${threshold}
done