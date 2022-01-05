get_make_cmd () {
    CMD="MSC_O_SAMPLING_RATIO=${MSC_O_SAMPLING_RATIO} TRAIN_SNT_NUM=${TRAIN_SNT_NUM} O_SAMPLING_RATIO=${O_SAMPLING_RATIO} POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${POSITIVE_RATIO_THR_OF_NEGATIVE_CAT} NEGATIVE_CATS=\"${NEGATIVE_CATS}\" WITH_O=${WITH_O} CHUNKER=${CHUNKER} make"
    echo ${CMD}
}

# Use all train snt
NEGATIVE_CATS=""
WITH_O=True
CHUNKER="enumerated"
POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=1.0
O_SAMPLING_RATIO=1.0
MSC_O_SAMPLING_RATIO=0.3



TRAIN_SNT_NUM=100
MSC_O_SAMPLING_RATIO=0.01
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)


TRAIN_SNT_NUM=500
MSC_O_SAMPLING_RATIO=0.0125
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)

TRAIN_SNT_NUM=750
MSC_O_SAMPLING_RATIO=0.02
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)

TRAIN_SNT_NUM=800
MSC_O_SAMPLING_RATIO=0.02125
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)

TRAIN_SNT_NUM=900
MSC_O_SAMPLING_RATIO=0.025
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)

TRAIN_SNT_NUM=1000
MSC_O_SAMPLING_RATIO=0.03
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)


TRAIN_SNT_NUM=10000
MSC_O_SAMPLING_RATIO=0.25
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)


TRAIN_SNT_NUM=9223372036854775807
MSC_O_SAMPLING_RATIO=0.3
MAKE=`get_make_cmd`
eval ${MAKE} train_on_gold -j$(nproc)