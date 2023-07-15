#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

eval_datasets=(CoNLL2003 MedMentions)
epoch_nums=(10 15 20 25 30)

for EVAL_DATASET in ${eval_datasets[@]}; do
    OUTPUT_DIR=outputs/${EVAL_DATASET}/gold/single
    mkdir -p ${OUTPUT_DIR}
    pwd >> ${OUTPUT_DIR}/cout
    ls -la >> ${OUTPUT_DIR}/cout
    for epoch_num in ${epoch_nums[@]}; do
        echo "epoch_num: ${epoch_num}" >>${OUTPUT_DIR}/cout
        MAKE="TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 NUM_TRAIN_EPOCHS=${epoch_num} EVAL_DATASET=${EVAL_DATASET} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
        eval ${MAKE} train_on_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
    done
done
