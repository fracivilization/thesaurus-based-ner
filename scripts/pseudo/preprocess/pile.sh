#!/bin/bash

# for file in data/raw/pile_type/*.txt; do
#   base=$(basename "$file" .txt)
#   echo "Processing $base"
#   echo "rye run python -m cli.preprocess.load_raw_corpus --source-txt data/raw/pile_type/${base}.txt --output-dir data/raw/${base}.dataset --spacy-model en_core_web_sm"
#   rye run python -m cli.preprocess.load_raw_corpus --source-txt data/raw/pile_type/${base}.txt --output-dir data/raw/${base}.dataset --spacy-model en_core_web_sm
#   echo "rye run python -m cli.preprocess.load_pseudo_multi_label_ner \
#       +multi_label_ner_model/multi_label_typer=CoNLL2003DictMatch \
#       ++multi_label_ner_model.multi_label_typer.term2cats=data/term2cats_with_common_sense \
#       ++multi_label_ner_model.remove_null_chunk=True \
#       +raw_corpus=data/raw/${base}.dataset \
#       +output_dir=data/pseudo/${base}"
#   rye run python -m cli.preprocess.load_pseudo_multi_label_ner \
#       +multi_label_ner_model/multi_label_typer=CoNLL2003DictMatch \
#       ++multi_label_ner_model.multi_label_typer.term2cats=data/term2cats_with_common_sense \
#       ++multi_label_ner_model.remove_null_chunk=True \
#       +raw_corpus=data/raw/${base}.dataset \
#       +output_dir=data/pseudo/${base}
# done

echo "rye run python -m cli.preprocess.load_raw_corpus --source-txt data/raw/pile_all.txt --output-dir data/raw/pile_all.dataset --spacy-model en_core_web_sm"
rye run python -m cli.preprocess.load_raw_corpus --source-txt data/raw/pile_all.txt --output-dir data/raw/pile_all.dataset --spacy-model en_core_sci_sm
echo "rye run python -m cli.preprocess.load_pseudo_multi_label_ner \
    +multi_label_ner_model/multi_label_typer=CoNLL2003DictMatch \
    ++multi_label_ner_model.multi_label_typer.term2cats=data/term2cats_with_common_sense \
    ++multi_label_ner_model.remove_null_chunk=True \
    +raw_corpus=data/raw/pile_all.dataset \
    +output_dir=data/pseudo/pile_all"
rye run python -m cli.preprocess.load_pseudo_multi_label_ner \
    +multi_label_ner_model/multi_label_typer=MedMentionsDictMatch \
    ++multi_label_ner_model.multi_label_typer.term2cats=data/term2cats_with_common_sense \
    ++multi_label_ner_model.remove_null_chunk=True \
    +raw_corpus=data/raw/pile_all.dataset \
    +output_dir=data/pseudo/pile_all