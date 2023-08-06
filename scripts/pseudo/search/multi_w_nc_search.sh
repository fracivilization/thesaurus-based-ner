epoch_nums=(5 10 15 20 25 30)
negative_ratios=(0.005 0.01 0.05 0.1 0.3 0.5 1.0 4.0 8.0 16.0)
for EPOCH_NUM in ${epoch_nums[@]}; do
    for NEGATIVE_RATIO in ${negative_ratios[@]}; do
        qsub -v EPOCH_NUM=${EPOCH_NUM} -v NEGATIVE_RATIO=${NEGATIVE_RATIO} scripts/pseudo/multi_w_nc.sh
    done
done