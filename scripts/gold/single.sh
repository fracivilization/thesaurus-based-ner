#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2305
dir=`dirname $0`

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

LD_LIBRARY_PATH=/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:
# EVAL_DATASET=CoNLL2003
# EVAL_DATASET=MedMentions
# early_stop_patiences=(3 5 7 10)
# eval_steps=(10 20 50 100)

OUTPUT_DIR=outputs/${EVAL_DATASET}/gold/single/num_train_epochs_${NUM_TRAIN_EPOCHS}
mkdir -p ${OUTPUT_DIR}
pwd >> ${OUTPUT_DIR}/cout
ls -la >> ${OUTPUT_DIR}/cout
echo "num_train_epochs: ${NUM_TRAIN_EPOCHS}" >>${OUTPUT_DIR}/cout
MAKE="TRAIN_BATCH_SIZE=8 EVAL_BATCH_SIZE=16 NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} EVAL_DATASET=${EVAL_DATASET} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
eval ${MAKE} train_on_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
