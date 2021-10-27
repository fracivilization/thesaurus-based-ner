make all -j$(nproc)
PSEUDO_DATA_ON_GOLD=$(make -n all | grep PSEUDO_DATA_ON_GOLD | awk '{print $3}')
RUN_OUT=$(python -m cli.train ++dataset.name_or_path=${PSEUDO_DATA_ON_GOLD})
echo $RUN_OUT
RUN_ID_BASE=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_BASE" ${RUN_ID_BASE}
FP_REMOVED_PSEUDO_DATA=$(make -n all | grep FP_REMOVED_PSEUDO_DATA | awk '{print $3}')
RUN_OUT=$(python -m cli.train ++dataset.name_or_path=${FP_REMOVED_PSEUDO_DATA})
echo $RUN_OUT
RUN_ID_WO_FP=$(echo $RUN_OUT | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_WO_FP" ${RUN_ID_WO_FP}

python -m cli.compare_metrics --base-run-id ${RUN_ID_BASE} --focus-run-id ${RUN_ID_WO_FP}