#$ -S /bin/bash
#$ -cwd
#$ -jc gpub-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
# EVAL_DATASET=CoNLL2003
EVAL_DATASET=MedMentions
OUTPUT_DIR=outputs/${EVAL_DATASET}/pseudo/multi_w_nc/preprocess
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
negative_ratios=(1.0 4.0 8.0 16.0)
for EPOCH_NUM in ${epoch_nums[@]}; do
    for NEGATIVE_RATIO in ${negative_ratios[@]}; do
        echo "epoch_num: ${EPOCH_NUM}, negative_ratio: ${NEGATIVE_RATIO}" >>${OUTPUT_DIR}/cout
        MAKE="EVAL_DATASET=${EVAL_DATASET} NUM_TRAIN_EPOCHS=${EPOCH_NUM} MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${NEGATIVE_RATIO} WITH_NEGATIVE_CATEGORIES=True make"
        eval ${MAKE} make_pseudo_multi_label_multi_classification_data_on_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
    done
done