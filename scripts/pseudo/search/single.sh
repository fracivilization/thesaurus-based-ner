# eval_datasets=("CoNLL2003" "MedMentions")
# negative_ratios=(0.1 0.5 1.0 2.0 3.0 4.0 8.0 16.0)

# for EVAL_DATASET in ${eval_datasets[@]}; do
#     for NEGATIVE_RATIO in ${negative_ratios[@]}; do
#         qsub -v NEGATIVE_RATIO=${NEGATIVE_RATIO} -v EVAL_DATASET=${EVAL_DATASET} scripts/pseudo/single.sh
#     done
# done

EVAL_DATASET="CoNLL2003"
negative_ratios=(0.3 3.0)
for NEGATIVE_RATIO in ${negative_ratios[@]}; do
    qsub -v NEGATIVE_RATIO=${NEGATIVE_RATIO} -v EVAL_DATASET=${EVAL_DATASET} scripts/pseudo/single.sh
done

EVAL_DATASET="MedMentions"
negative_ratios=(0.3 32.0 64.0)
for NEGATIVE_RATIO in ${negative_ratios[@]}; do
    qsub -v NEGATIVE_RATIO=${NEGATIVE_RATIO} -v EVAL_DATASET=${EVAL_DATASET} scripts/pseudo/single.sh
done