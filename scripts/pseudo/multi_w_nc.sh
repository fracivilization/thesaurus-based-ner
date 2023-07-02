#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
EVAL_DATASET=CoNLL2003
OUTPUT_DIR=outputs/${EVAL_DATASET}/pseudo/multi_w_nc
mkdir -p ${OUTPUT_DIR}
pwd >> ${OUTPUT_DIR}/cout
ls -la >> ${OUTPUT_DIR}/cout

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL


epoch_nums=(5 10 15 20 25 30)
negative_ratios=(0.005 0.01 0.05 0.1 0.3 0.5 1.0 4.0 8.0 16.0)
for epoch_num in ${epoch_nums[@]}; do
    for negative_ratio in ${negative_ratios[@]}; do
        echo "epoch_num: ${epoch_num}, negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
        MAKE="EVAL_DATASET=${EVAL_DATASET} NUM_TRAIN_EPOCHS=${epoch_num} MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} WITH_NEGATIVE_CATEGORIES=True MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE=True make"
        eval ${MAKE} eval_flatten_marginal_softmax -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
    done
done
