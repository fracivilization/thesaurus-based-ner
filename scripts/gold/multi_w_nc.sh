#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
OUTPUT_DIR=outputs/search_params/negative_positive_ratio_on_gold_w_nc
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

negative_ratios=(1.0 4.0 16.0 64.0 128.0)
# negative_ratios=(32.0 64.0 128.0)
for negative_ratio in ${negative_ratios[@]}; do
    echo "negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
    MAKE="WITH_NEGATIVE_CATEGORIES=True MSMLC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
    eval ${MAKE} eval_flatten_marginal_softmax_gold -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
    # eval ${MAKE} -n train_flattern_multilabel_ner_gold
done
# echo ${MAKEOPT}
