#!/bin/bash
dir=`dirname $0`
source ${dir}/../params.sh
MAKEOPT=${make_opts[4]} 
thresholds=(0.002 0.004 0.006 0.008)
CHUNKER="spacy_np"
for threshold in ${thresholds[@]}; do
    MAKEOPT="${make_opts[4]} POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${threshold}"
    echo ${MAKEOPT}
    # TODO: Makefileからterm2catのパスを取得する
    eval "${MAKEOPT} make term2cat"
    TERM2CAT=`eval "${MAKEOPT} make -n term2cat" | grep -m1 TERM2CAT | awk '{print $3}'`
    echo ${TERM2CAT}
    # eval ${MAKEOPT} make term2cat
    # TODO: 取得されたterm2catを利用して行う擬似アノテーションの精度を表示する
    echo "threshold: $threshold"
    echo "poetry run python -m cli.train \
        ner_model=PseudoTwoStage ner_model/typer/term2cat=${TERM2CAT} \
        ner_model.typer.term2cat=${TERM2CAT}"
    poetry run python -m cli.train \
        ner_model=PseudoTwoStage ner_model.typer.term2cat=${TERM2CAT} \
        testor.baseline_typer.term2cat=${TERM2CAT}
done
# echo ${MAKEOPT}
