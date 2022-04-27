#!/bin/bash
dir=`dirname $0`
negative_ratios=(1.0 4.0 16.0 64.0 128.0)
# negative_ratios=(32.0 64.0 128.0)
for negative_ratio in ${negative_ratios[@]}; do
    MAKE="NEGATIVE_CATS=\"T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200\" MSMLC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} eval_flatten_marginal_softmax_gold -j$(nproc)
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
