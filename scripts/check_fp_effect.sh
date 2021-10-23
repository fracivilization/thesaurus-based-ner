# TODO: 学習データを追加する
PSEUDO_DATA_ON_GOLD=$(make -n all | grep PSEUDO_DATA_ON_GOLD | awk '{print $3}')
RUN_OUT=$(python -m cli.train ++dataset.name_or_path=${PSEUDO_DATA_ON_GOLD})
RUN_ID_BASE=$(echo $TRAIN_OUT | grep "mlflow_run_id" | awk -F " " '{print $2}')
# TODO: 学習データを追加する
FP_REMOVED_PSEUDO_DATA=$(make -n all | grep FP_REMOVED_PSEUDO_DATA | awk '{print $3}')
RUN_OUT=$(python -m cli.train ++dataset.name_or_path=${FP_REMOVED_PSEUDO_DATA})
RUN_ID_WO_FP=$(echo $TRAIN_OUT | grep "mlflow_run_id" | awk -F " " '{print $2}')

