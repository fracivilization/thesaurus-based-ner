#!/bin/bash
dir=`dirname $0`
source ${dir}/params.sh

TMPFILE=$(mktemp)

echo "PseudoAnno"
WITH_NC=True
GOLD_NER_DATA_DIR=$(WITH_NC=${WITH_NC} make -n all | grep GOLD_NER_DATA_DIR | awk '{print $3}')
poetry run python -m cli.train \
    ner_model=PseudoTwoStage \
    ++dataset.name_or_path=${GOLD_NER_DATA_DIR} 2>&1 | tee ${TMPFILE}
RUN_ID_PseudoAnno=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_PseudoAnno" ${RUN_ID_PseudoAnno}

get_enumerated_model_cmd () {
    CMD="\
        poetry run python -m cli.train \
        ++dataset.name_or_path=${GOLD_NER_DATA} \
        ner_model.typer.msc_datasets=${PSEUDO_MSC_DATA}\
        ner_model/chunker=${FIRST_STAGE_CHUNKER} \
        ner_model.typer.model_args.o_sampling_ratio=${O_SAMPLING_RATIO} \
        ner_model.typer.train_args.per_device_train_batch_size=8 \
        ner_model.typer.train_args.per_device_eval_batch_size=32 \
        ner_model.typer.train_args.do_train=True \
        ner_model.typer.train_args.overwrite_output_dir=True \
        testor.baseline_typer.term2cat=${TERM2CAT}
    "
    echo ${CMD}
}
get_make_cmd () {
    CMD="TRAIN_SNT_NUM=${TRAIN_SNT_NUM} O_SAMPLING_RATIO=${O_SAMPLING_RATIO} POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT} NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=${WITH_O} FIRST_STAGE_CHUNKER=${FIRST_STAGE_CHUNKER} make"
    echo ${CMD}
}

# TODO: O_SAMPLING_RATIO に対してもループを回したいので、indexを指定してmake_optsからMAKEOPTSを取ってこれるようにする

# o_sampling_ratios=(0.0001 0.02 0.00 0.02 0.00)
# for i in `seq 0 $((${#make_opts[@]} - 1))`; do
#     MAKEOPTS=${make_opts[i]}
#     echo "MAKEOPTS: ${MAKEOPTS}"
#     eval ${MAKEOPTS} make all -j$(nproc)
#     RUN_DATASET=$(eval ${MAKEOPTS} make -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
#     O_SAMPLING_RATIO=${o_sampling_ratios[i]}
# done

TRAIN_SNT_NUM=9223372036854775807
echo "GOLD"
# Get Dataset
# All Negatives
NEGATIVE_CATS=""
WITH_O=True
FIRST_STAGE_CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=1.0
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold 2>&1 | tee ${TMPFILE}
RUN_ID_AllNegatives=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives" ${RUN_ID_AllNegatives}

echo "All Negatives"
# Get Dataset
# All Negatives
NEGATIVE_CATS=""
WITH_O=True
FIRST_STAGE_CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
# O_SAMPLING_RATIO=0.0001
O_SAMPLING_RATIO=0.03
MAKE=`get_make_cmd`
eval ${MAKE} train -j$(nproc)

# echo "All Negatives (NP)"
# NEGATIVE_CATS=""
# WITH_O=True
# FIRST_STAGE_CHUNKER="spacy_np"
# POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
# O_SAMPLING_RATIO=0.02
# MAKE=`get_make_cmd`
# eval ${MAKE} train 2>&1 | tee ${TMPFILE}
# RUN_ID_AllNegatives_NP=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
# echo "RUN_ID_AllNegatives (NP)" ${RUN_ID_AllNegatives_NP}

# echo "Thesaurus Negatives (UMLS)"
# NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
# WITH_O=False
# FIRST_STAGE_CHUNKER="spacy_np"
# POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
# O_SAMPLING_RATIO=0.00
# MAKE=`get_make_cmd`
# eval ${MAKE} train 2>&1 | tee ${TMPFILE}
# RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
# echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


# echo "Thesaurus Negatives (UMLS) + All Negatives (NP)"
# NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
# WITH_O=True
# FIRST_STAGE_CHUNKER="spacy_np"
# POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
# O_SAMPLING_RATIO=0.02
# MAKE=`get_make_cmd`
# CMD=`get_enumerated_model_cmd`
# eval ${MAKE} train 2>&1 | tee ${TMPFILE}
# RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
# echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


echo "Thesaurus Negatives (UMLS) + All Negatives"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
WITH_O=True
FIRST_STAGE_CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.03
O_SAMPLING_RATIO=0.005
MAKE=`get_make_cmd`
eval ${MAKE} train -j$(nproc) 
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


# # Thesaurus Negatives (UMLS + DBPedia)
# echo "Thesaurus Negatives (UMLS + DBPedia)"outputs/2022-02-11/08-13-06
# NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200 GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour"
# WITH_O=False
# FIRST_STAGE_CHUNKER="spacy_np"
# POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=0.1
# O_SAMPLING_RATIO=0.01
# MAKE=`get_make_cmd`
# eval ${MAKE} all -j$(nproc)
# eval ${MAKE} train -j$(nproc) 
# RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
# echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}

# Multi Label Gold

MAKE=" make"
eval ${MAKE} train_flattern_multilabel_ner_gold -j$(nproc)