#!/bin/bash
dir=`dirname $0`
# negative_ratios=(1.0 4.0 16.0 64.0 128.0)
# negative_ratios=(32.0 64.0 128.0)
negative_ratios=(6.0 8.0 10.0 12.0 14.0)


for negative_ratio in ${negative_ratios[@]}; do
    MAKE="NEGATIVE_CATS=\"\" WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0 NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} train_on_gold -j$(nproc)
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
