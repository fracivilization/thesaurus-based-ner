# Arguments
## 実行に利用するPythonのパス
PYTHON ?= .venv/bin/python
## 疑似教師として利用する知識ベース or アンカーテキスト
KNOWLEDGE_BASE ?= DBPedia# or UMLS or WikipediaAnchorText
## 評価に利用するデータセット
EVAL_DATASET ?= CoNLL2003# or MedMentions
## EVAL_DATASET==MedMentionsの時のデフォルトPOSITIVE_CATS
# POSITIVE_CATS ?= T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
## EVAL_DATASET==CoNLL2003の時のデフォルトPOSITIVE_CATS
POSITIVE_CATS ?= PER LOC ORG MISC
## 負例のカテゴリを追加して利用するかどうか
WITH_NEGATIVE_CATEGORIES ?= False
## 一つの単語に対して複数のentityが存在する場合に、複数のentityで共通するカテゴリ飲みを利用する
REMAIN_COMMON_SENSE_FOR_TERM2CATS ?= True # 着目カテゴリ鹿利用しない場合（通常のDS NERと同じ場合）は、term2cats計算時に複数のエンティティの持つ共通の意味に限定して利用しない
## 補完カテゴリ
NEGATIVE_CATS ?= "$(shell ${PYTHON} -m cli.preprocess.load_negative_categories --positive-categories $(subst $() ,_,$(POSITIVE_CATS)) --with-negative_categories $(WITH_NEGATIVE_CATEGORIES) --eval-dataset $(EVAL_DATASET))"
## Oラベルを使う
WITH_O ?= True
## ２段階モデルの１段階目 擬似データの際のChunkerを意味しない
FIRST_STAGE_CHUNKER ?= enumerated
## 正例に対する負例の比率
NEGATIVE_RATIO_OVER_POSITIVE ?= 1.0
## 訓練時に使う教師及び疑似教師データ数
TRAIN_SNT_NUM ?= 9223372036854775807
## MultiSpanMultiLabel設定でPositiveとNegativeの比率を合わせるためにundersamplingをするかどうか
### STATICは事前に比率を合わせておく方法で、
MSMLC_STATIC_PN_RATIO_EQUIVALENCE ?= False
### DYNAMICは訓練時に比率を合わせる方法
MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE ?= False
## MSMLC STATIC, DYNAMIC SAMPLINGで利用する負例の比率
MSMLC_NEGATIVE_RATIO_OVER_POSITIVE ?= 1.0
## 訓練時(TRAIN)と評価時(EVAL)のミニバッチサイズ
TRAIN_BATCH_SIZE ?= 16
EVAL_BATCH_SIZE ?= 32


## MultiSpanClassifierで利用する引数群
MSC_ARGS := "WITH_O: $(WITH_O) FIRST_STAGE_CHUNKER: $(FIRST_STAGE_CHUNKER) NEGATIVE_RATIO_OVER_POSITIVE: $(NEGATIVE_RATIO_OVER_POSITIVE)"
## MultiSpanMultiLabelClassifierで利用する引数群
MSMLC_ARGS := "FIRST_STAGE_CHUNKER: $(FIRST_STAGE_CHUNKER) MSMLC_STATIC_PN_RATIO_EQUIVALENCE: $(MSMLC_STATIC_PN_RATIO_EQUIVALENCE) MSMLC_NEGATIVE_RATIO_OVER_POSITIVE: $(MSMLC_NEGATIVE_RATIO_OVER_POSITIVE) MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE: $(MSMLC_DYNAMIC_PN_RATIO_EQUIVALENCE)"