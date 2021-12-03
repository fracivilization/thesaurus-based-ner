#!/bin/bash
dir=`dirname $0`
source ${dir}/../params.sh
MAKEOPT=${make_opts[4]} 
thresholds=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
CHUNKER="spacy_np"
for threshold in ${thresholds[@]}; do
    MAKEOPT="${make_opts[4]} POSITIVE_RATIO_THR_OF_NEGATIVE_CAT=${threshold}"
    echo ${MAKEOPT}
    # TODO: Makefileからterm2catのパスを取得する
    eval "${MAKEOPT} make term2cat"
    TERM2CAT=`eval "${MAKEOPT} make -n term2cat" | grep TERM2CAT | awk '{print $3}'`
    # eval ${MAKEOPT} make term2cat
    # TODO: 取得されたterm2catを利用して行う擬似アノテーションの精度を表示する
    # echo "threshold: $threshold"
    poetry run python -m cli.train \
        ner_model=PseudoTwoStage ner_model/typer/term2cat=${TERM2CAT} \
        ner_model.typer.term2cat=${TERM2CAT}
done
# echo ${MAKEOPT}
