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
        ner_model/chunker=${CHUNKER} \
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
    CMD="O_SAMPLING_RATIO=${O_SAMPLING_RATIO} POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT} NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=${WITH_O} CHUNKER=${CHUNKER} make"
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

echo "GOLD"
# Get Dataset
# All Negatives
NEGATIVE_CATS=""
WITH_O=True
CHUNKER="enumerated"
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
CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.0001
MAKE=`get_make_cmd`
RUN_DATASET=$(eval ${MAKE} -n all | grep PSEUDO_NER_DATA_DIR | awk '{print $3}')
CMD=`get_enumerated_model_cmd`
eval ${MAKE} train 2>&1 | tee ${TMPFILE}
RUN_ID_AllNegatives=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives" ${RUN_ID_AllNegatives}

echo "All Negatives (NP)"
NEGATIVE_CATS=""
WITH_O=True
CHUNKER="spacy_np"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.02
MAKE=`get_make_cmd`
eval ${MAKE} train 2>&1 | tee ${TMPFILE}
RUN_ID_AllNegatives_NP=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_AllNegatives (NP)" ${RUN_ID_AllNegatives_NP}

echo "Thesaurus Negatives (UMLS)"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
WITH_O=False
CHUNKER="spacy_np"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.00
MAKE=`get_make_cmd`
eval ${MAKE} train 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


echo "Thesaurus Negatives (UMLS) + All Negatives (NP)"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
WITH_O=True
CHUNKER="spacy_np"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.02
MAKE=`get_make_cmd`
CMD=`get_enumerated_model_cmd`
eval ${MAKE} train 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


echo "Thesaurus Negatives (UMLS) + All Negatives"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200"
WITH_O=True
CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=0.0001
MAKE=`get_make_cmd`
eval ${MAKE} train 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}


# Thesaurus Negatives (UMLS + DBPedia)
echo "Thesaurus Negatives (UMLS + DBPedia)"
NEGATIVE_CATS="T054 T055 T056 T064 T065 T066 T068 T075 T079 T080 T081 T099 T100 T101 T102 T171 T194 T200 GeneLocation Species Disease Work SportsSeason Device Media SportCompetitionResult EthnicGroup Protocol Award Demographics MeanOfTransportation FileSystem Medicine Area Flag UnitOfWork MedicalSpecialty GrossDomesticProduct Biomolecule Identifier Blazon PersonFunction List TimePeriod Event Relationship Altitude TopicalConcept Spreadsheet Currency Cipher Browser Tank Food Depth Population Statistic StarCluster Language GrossDomesticProductPerCapita ChemicalSubstance ElectionDiagram Diploma Place Algorithm ChartsPlacements Unknown Activity PublicService Agent Name AnatomicalStructure Colour"
WITH_O=False
CHUNKER="spacy_np"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=0.1
O_SAMPLING_RATIO=0.02
MAKE=`get_make_cmd`
eval ${MAKE} all -j$(nproc)
eval ${MAKE} train -j$(nproc) 2>&1 | tee ${TMPFILE}
RUN_ID_Thesaurus_Negatives_UMLS=$(cat ${TMPFILE} | grep "mlflow_run_id" | awk '{print $2}')
echo "RUN_ID_Thesaurus_Negatives (UMLS)" ${RUN_ID_Thesaurus_Negatives_UMLS}