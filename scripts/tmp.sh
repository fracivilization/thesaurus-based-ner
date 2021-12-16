#!/bin/bash
dir=`dirname $0`
source ${dir}/params.sh

o_sampling_ratios=(0.0001 0.02 0.00 0.02 0.00)
for i in `seq 0 $((${#make_opts[@]} - 1))`; do
    MAKEOPTS=${make_opts[i]}
    echo "MAKEOPTS: ${MAKEOPTS}"
    # eval ${MAKEOPTS} make all -j$(nproc)
    RUN_DATASET=$(eval ${MAKEOPTS} make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
    O_SAMPLING_RATIO=${o_sampling_ratios[i]}
    echo ${O_SAMPLING_RATIO}
done