#$ -S /bin/bash
#$ -cwd
#$ -jc gpub-container_g4
#$ -ac d=nvcr-pytorch-2305
dir=`dirname $0`

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

# TODO: 連想配列使って両方の実験設定を回せるようにする
LD_LIBRARY_PATH=/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:/home/takayo-s/.linuxbrew/Cellar/libffi/3.4.4/lib/:/home/takayo-s/.linuxbrew/Cellar/openssl@1.1/1.1.1q/lib/:/home/takayo-s/.linuxbrew/Cellar/libx11/1.8.1/lib:

OUTPUT_DIR=outputs/${EVAL_DATASET}/pseudo/single/negative_ratio=${NEGATIVE_RATIO}
mkdir -p ${OUTPUT_DIR}
pwd >> ${OUTPUT_DIR}/cout
ls -la >> ${OUTPUT_DIR}/cout
echo "negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
MAKE="EVAL_DATASET=${EVAL_DATASET} REMAIN_COMMON_SENSE_FOR_TERM2CATS=False NEGATIVE_RATIO_OVER_POSITIVE=${NEGATIVE_RATIO} make"
eval ${MAKE} train -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
