#!/bin/bash
dir=`dirname $0`
# negative_ratios=(1.0 2.0 4.0 8.0 16.0)
negative_ratios=(0.9 0.7 0.5 0.3 0.1)
for negative_ratio in ${negative_ratios[@]}; do
    MAKE="NEGATIVE_CATS=\"\" MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} eval_flatten_marginal_softmax -j$(nproc)
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
