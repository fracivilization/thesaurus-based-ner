#!/bin/bash
dir=`dirname $0`
source ${dir}/params.sh
for MAKEOPTS in "${make_opts[@]}"; do
    echo "MAKEOPTS: ${MAKEOPTS}"
    eval ${MAKEOPTS} make all -j$(nproc)
done