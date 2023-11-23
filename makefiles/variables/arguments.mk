# Arguments
## 実行に利用するPythonのパス
PYTHON ?= .venv/bin/python
## 評価に利用するデータセット
EVAL_DATASET ?= CoNLL2003# or MedMentions
# EVAL_DATASET ?= MedMentions
CoNLL2003_POSITIVE_CATS ?= PER LOC ORG MISC
MedMentions_POSITIVE_CATS ?= T005 T007 T017 T022 T031 T033 T037 T038 T058 T062 T074 T082 T091 T092 T097 T098 T103 T168 T170 T201 T204
POSITIVE_CATS ?= $($(EVAL_DATASET)_POSITIVE_CATS)
## 負例のカテゴリを追加して利用するかどうか
WITH_NEGATIVE_CATEGORIES ?= False
## 一つの単語に対して複数のentityが存在する場合に、複数のentityで共通するカテゴリ飲みを利用する
REMAIN_COMMON_SENSE_FOR_TERM2CATS ?= True # 着目カテゴリ鹿利用しない場合（通常のDS NERと同じ場合）は、term2cats計算時に複数のエンティティの持つ共通の意味に限定して利用しない
## Oラベルを使う
WITH_O ?= True
## ２段階モデルの１段階目 擬似データの際のChunkerを意味しない
FIRST_STAGE_CHUNKER ?= enumerated
## 正例に対する負例の比率
NEGATIVE_RATIO_OVER_POSITIVE ?= 1.0
## 訓練時に使う教師及び疑似教師データ数
TRAIN_SNT_NUM ?= 9223372036854775807
## MSMLC STATIC, DYNAMIC SAMPLINGで利用する負例の比率
MSMLC_NEGATIVE_RATIO_OVER_POSITIVE ?= 1.0
## 訓練時(TRAIN)と評価時(EVAL)のミニバッチサイズ
TRAIN_BATCH_SIZE ?= 12
EVAL_BATCH_SIZE ?= 24
## Few-Shot設定に使うShot数
FEW_SHOT_NUM ?= 5
## 訓練のepoch数 (基本変えないがFew-Shot学習時のみ変更する)
NUM_TRAIN_EPOCHS ?= 40 # Early Stoppingの場合は最大epoch数
EARLY_STOPPING_PATIENCE ?= 5
EVAL_STEPS ?= 100



# 中間変数
## 疑似教師として利用する知識ベース or アンカーテキスト
MedMentions_KNOWLEDGE_BASE := UMLS
CoNLL2003_KNOWLEDGE_BASE := DBPedia# or WikipediaAnchorText
KNOWLEDGE_BASE := $($(EVAL_DATASET)_KNOWLEDGE_BASE)# or UMLS or WikipediaAnchorText
## 補完カテゴリ
NEGATIVE_CATS := "$(shell ${PYTHON} -m cli.preprocess.load_negative_categories --positive-categories "$(subst $() ,_,$(POSITIVE_CATS))" --with-negative_categories $(WITH_NEGATIVE_CATEGORIES) --eval-dataset $(EVAL_DATASET))"
## 事前学習モデル名
MODEL_NAME := $($(EVAL_DATASET)_MODEL_NAME)
## MultiSpanClassifierで利用する引数群
MSC_ARGS := "WITH_O: $(WITH_O) FIRST_STAGE_CHUNKER: $(FIRST_STAGE_CHUNKER) NEGATIVE_RATIO_OVER_POSITIVE: $(NEGATIVE_RATIO_OVER_POSITIVE) EARLY_STOPPING_PATIENCE: $(EARLY_STOPPING_PATIENCE)"
## MultiSpanMultiLabelClassifierで利用する引数群
MSMLC_ARGS := "FIRST_STAGE_CHUNKER: $(FIRST_STAGE_CHUNKER) MSMLC_NEGATIVE_RATIO_OVER_POSITIVE: $(MSMLC_NEGATIVE_RATIO_OVER_POSITIVE) EARLY_STOPPING_PATIENCE: $(EARLY_STOPPING_PATIENCE)"


# CHECK ARGUMENTS
check_positive_cats:
	echo ${POSITIVE_CATS}
check_negative_cats:
	echo ${NEGATIVE_CATS}