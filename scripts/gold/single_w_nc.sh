#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
OUTPUT_DIR=outputs/gold/single_w_nc
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

negative_ratios=(5.0 6.0 7.0 8.0 9.0)


for negative_ratio in ${negative_ratios[@]}; do
    echo "negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
    MAKE="WITH_NEGATIVE_CATEGORIES=True WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} train_on_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
done
