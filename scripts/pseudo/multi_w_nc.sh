#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
# EVAL_DATASET=CoNLL2003
EVAL_DATASET=MedMentions
OUTPUT_DIR=outputs/${EVAL_DATASET}/pseudo/multi_w_nc/negative_ratio=${NEGATIVE_RATIO}/epoch_num=${EPOCH_NUM}
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


echo "epoch_num: ${EPOCH_NUM}, negative_ratio: ${NEGATIVE_RATIO}" >>${OUTPUT_DIR}/cout
MAKE="EVAL_DATASET=${EVAL_DATASET} NUM_TRAIN_EPOCHS=${EPOCH_NUM} MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${NEGATIVE_RATIO} WITH_NEGATIVE_CATEGORIES=True make"
eval ${MAKE} eval_flatten_marginal_softmax -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
