#$ -S /bin/bash
#$ -cwd
#$ -jc gpub-container_g4
#$ -ac d=nvcr-pytorch-2305
dir=`dirname $0`
LD_LIBRARY_PATH=/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:
# EVAL_DATASET=CoNLL2003
# EVAL_DATASET=MedMentions
NUM_TRAIN_EPOCHS=40
OUTPUT_DIR=outputs/${EVAL_DATASET}/gold/low_resource/num_train_epochs_${NUM_TRAIN_EPOCHS}
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

DIR=${OUTPUT_DIR}/${TRAIN_SNT_NUM}/${RANDOM_SEED}
mkdir -p ${DIR}
eval_step=1
echo "train_snt_num: ${TRAIN_SNT_NUM} eval_step: ${eval_step}" >>${OUTPUT_DIR}/cout
# MAKE="EVAL_STEPS=${eval_step} TRAIN_SNT_NUM=${TRAIN_SNT_NUM} EVAL_DATASET=${EVAL_DATASET} EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
MAKE="TRAIN_SNT_NUM=${TRAIN_SNT_NUM} EVAL_DATASET=${EVAL_DATASET} NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS} RANDOM_SEED=${RANDOM_SEED} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" make"
eval ${MAKE} train_on_gold -j$(nproc) >>${DIR}/cout 2>>${DIR}/cerr
