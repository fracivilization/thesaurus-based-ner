#!/bin/bash
dir=`dirname $0`
negative_ratios=(0.1 0.5 1,0)
# negative_ratios=(32.0 64.0 128.0)
for negative_ratio in ${negative_ratios[@]}; do
    MAKE="NEGATIVE_CATS=\"\" MSMLC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} eval_flatten_marginal_softmax_gold -j$(nproc)
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
