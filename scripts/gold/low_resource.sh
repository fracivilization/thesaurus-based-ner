#$ -S /bin/bash
#$ -cwd
#$ -jc gpub-container_g4
#$ -ac d=nvcr-pytorch-2305
dir=`dirname $0`
LD_LIBRARY_PATH=/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:
EVAL_DATASET=CoNLL2003
# EVAL_DATASET=MedMentions
OUTPUT_DIR=outputs/${EVAL_DATASET}/gold/low_resource
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

train_snt_nums=(10 50 100 200 400 500 600 700 800 900 1000 1200 1500 2000 2500 3000 4000 5000 6000 7000)

for train_snt_num in ${train_snt_nums[@]}; do
    DIR=${OUTPUT_DIR}/${train_snt_num}
    mkdir -p ${DIR}
    echo "train_snt_num: ${train_snt_num}" >>${OUTPUT_DIR}/cout
    MAKE="TRAIN_SNT_NUM=${train_snt_num} EVAL_DATASET=${EVAL_DATASET} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
    eval ${MAKE} train_on_gold -j$(nproc) >>${DIR}/cout 2>>${DIR}/cerr
done
