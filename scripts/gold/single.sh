#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
EVAL_DATASET=CoNLL2003
OUTPUT_DIR=outputs/${EVAL_DATASET}/gold/single
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

MAKE="EVAL_DATASET=${EVAL_DATASET} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
eval ${MAKE} train_on_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
