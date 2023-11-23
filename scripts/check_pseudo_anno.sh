EVAL_DATASET="CoNLL2003"
qsub -v EVAL_DATASET=${EVAL_DATASET} scripts/pseudo/pseudo_anno.sh

EVAL_DATASET="MedMentions"
qsub -v EVAL_DATASET=${EVAL_DATASET} scripts/pseudo/pseudo_anno.sh