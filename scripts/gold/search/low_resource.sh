eval_datasets=("CoNLL2003" "MedMentions")
train_snt_nums=(10 50 100 200 400 500 600 700 800 900 1000 1200 1500 2000 2500 3000 4000 5000 6000 7000)

EVAL_DATASET="CoNLL2003"
# train_snt_nums=(100 200 500)
for TRAIN_SNT_NUM in ${train_snt_nums[@]}; do
    qsub -v EVAL_DATASET=${EVAL_DATASET} -v TRAIN_SNT_NUM=${TRAIN_SNT_NUM} scripts/gold/low_resource.sh
done

EVAL_DATASET="MedMentions"
for TRAIN_SNT_NUM in ${train_snt_nums[@]}; do
    qsub -v EVAL_DATASET=${EVAL_DATASET} -v TRAIN_SNT_NUM=${TRAIN_SNT_NUM} scripts/gold/low_resource.sh
done