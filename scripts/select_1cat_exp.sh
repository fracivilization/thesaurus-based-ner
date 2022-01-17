#!/bin/bash
ST21pv=(T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204)
ST21pv=(T005 T007 T017 T022 T031 T033 T037 T038 T058 T062)
# ST21pv=(T062)
dir=`dirname $0`
# source ${dir}/params.sh

get_make () {
    FOCUS_CATS=$1
    NEGATIVE_CATS=`poetry run python -m cli.get_umls_negative_cats --focus-cats ${FOCUS_CATS}`
    local make_opts=(
        "FOCUS_CATS=${FOCUS_CATS} NEGATIVE_CATS=\"\" CHUNKER=spacy_np POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0 make train_pseudo_anno"
        "FOCUS_CATS=${FOCUS_CATS} NEGATIVE_CATS=\"\" WITH_O=True CHUNKER=enumerated POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0 O_SAMPLING_RATIO=0.03 make train"
        "FOCUS_CATS=${FOCUS_CATS} NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=True CHUNKER=enumerated POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0 O_SAMPLING_RATIO=0.005 make train"
    )
    for line in ${make_opts[@]}; do
        echo $line
    done
}

for cat in ${ST21pv[@]}; do
    echo $cat
    OLD_IFS=$IFS
    IFS=$'\n'
    get_make ${cat} | while read MAKE; do
        echo $MAKE 
        eval ${MAKE} -j$(nproc)
        # eval ${MAKE} -n 
    done
    IFS=$OLD_IFS
done
