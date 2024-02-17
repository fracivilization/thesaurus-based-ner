eval_datasets=("CoNLL2003" "MedMentions")
train_snt_nums=(10 50 100 200 400 500 600 700 800 900 1000 1200 1500 2000 2500 3000 4000 5000 6000 7000)
RANDOM_ITER_NUM=10
# EVAL_DATASET="CoNLL2003"
# train_snt_nums=(100 200 500)
# train_snt_nums=()

for EVAL_DATASET in ${eval_datasets[@]}; do
    for TRAIN_SNT_NUM in ${train_snt_nums[@]}; do
        for i in {1..${RANDOM_ITER_NUM}}; do
            qsub -v EVAL_DATASET=${EVAL_DATASET} -v TRAIN_SNT_NUM=${TRAIN_SNT_NUM} -v RANDOM_SEED=${RANDOM} scripts/gold/low_resource.sh
        done
    done
done

# EVAL_DATASET="MedMentions"
# train_snt_nums=(1400)
# for TRAIN_SNT_NUM in ${train_snt_nums[@]}; do
#     qsub -v EVAL_DATASET=${EVAL_DATASET} -v TRAIN_SNT_NUM=${TRAIN_SNT_NUM} scripts/gold/low_resource.sh
# done
