#$ -S /bin/bash
#$ -cwd
#$ -jc gpu-container_g4
#$ -ac d=nvcr-pytorch-2205
dir=`dirname $0`
OUTPUT_DIR=outputs/pseudo/multi_w_latent
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

negative_ratios=(0.5 0.3 0.1 0.05 0.01 0.005)
latent_label_dims=(10 30 50 70 90)
weight_for_latent_labels=(0.1 0.3 0.5)
for weight in ${weight_for_latent_labels[@]}; do
    for latent_label_dim in ${latent_label_dims[@]}; do
        for negative_ratio in ${negative_ratios[@]}; do
            echo "weight: ${weight} latent_label_dim: ${latent_label_dim} negative_ratio: ${negative_ratio}" >>${OUTPUT_DIR}/cout
            MAKE="USE_LATENT_LABEL=True LATENT_LABEL_DIM=${latent_label_dim} WEIGHT_FOR_LATENT_LABEL=${weight} WITH_NEGATIVE_CATEGORIES=True MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE=True MSMLC_NEGATIVE_RATIO_OVER_POSITIVE=${negative_ratio} make"
            eval ${MAKE} eval_flatten_marginal_softmax -j$(nproc) >>${OUTPUT_DIR}/cout 2>>${OUTPUT_DIR}/cerr
        done
    done
done
