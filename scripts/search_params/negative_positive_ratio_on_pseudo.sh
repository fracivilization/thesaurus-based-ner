#!/bin/bash
dir=`dirname $0`
negative_ratios=(1.0 2.0 4.0 8.0 16.0)
# negative_ratios=(32.0 64.0 128.0)
for negative_ratio in ${negative_ratios[@]}; do
    MAKE="MSMLC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} train_and_eval_flatten_marginal_softmax -j$(nproc)
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
