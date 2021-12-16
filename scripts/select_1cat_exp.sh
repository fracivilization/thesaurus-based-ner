#!/bin/bash
ST21pv=(T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204)
dir=`dirname $0`
source ${dir}/params.sh
for cat in ${ST21pv[@]}; do
    echo $cat
    NEGATIVE_CATS=`poetry run python -m cli.get_umls_negative_cat ${cat}`
    make_opts=`get_make_opts ${NEGATIVE_CATS}`
    OLD_IFS=$IFS
    IFS=$'\n'
    get_make_opts ${NEGATIVE_CATS} | while read MAKEOPT; do
        echo $MAKEOPT
        # eval ${MAKEOPTS} make all -j$(nproc)
        eval ${MAKEOPTS} make -n all
    done
    IFS=$OLD_IFS
done