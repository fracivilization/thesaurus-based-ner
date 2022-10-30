#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
OUTPUT_DIR=outputs/pseudo/single_w_nc
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

negative_ratios=(0.02 0.03 0.04 0.05)
for negative_ratio in ${negative_ratios[@]}; do
    echo "negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
    MAKE="WITH_NEGATIVE_CATEGORIES=True NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} WITH_O=True FIRST_STAGE_CHUNKER=\"enumerated\" POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0 make"
    eval ${MAKE} train -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
done
