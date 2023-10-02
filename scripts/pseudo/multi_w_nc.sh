#$ -S /bin/bash
#$ -cwd
#$ -jc gpub-container_g4
#$ -ac d=nvcr-pytorch-2305
dir=`dirname $0`
EVAL_DATASET=CoNLL2003
# EVAL_DATASET=MedMentions
LD_LIBRARY_PATH=/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:
EVAL_STEPS=100
OUTPUT_DIR=outputs/${EVAL_DATASET}/pseudo/multi_w_nc/negative_ratio=${NEGATIVE_RATIO}/eval_steops=${EVAL_STEPS}
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

echo "negative_ratio: ${NEGATIVE_RATIO}" >>${OUTPUT_DIR}/cout
MAKE="EVAL_DATASET=${EVAL_DATASET} MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${NEGATIVE_RATIO} EVAL_STEPS=${EVAL_STEPS} WITH_NEGATIVE_CATEGORIES=True make"
eval ${MAKE} eval_flatten_marginal_softmax -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
# TODO: そもそも生成させないようにする
rm -r `find outputs -type d -regex '.*/checkpoint-[0-9]+'`
