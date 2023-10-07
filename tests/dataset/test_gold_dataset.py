from src.dataset.gold_dataset import load_gold_datasets

CoNLL_MULTI_LABEL_NER_DATASETS = "tests/fixtures/mini_conll_multi_label_ner_dataset"
MEDMENTIONS_MULTI_LABEL_NER_DATASETS = "tests/fixtures/mini_medmentions_multi_label_ner_dataset"

def test_load_gold_conll_dataset():
    positive_cats = "PER_LOC_ORG_MISC"
    negative_cats = ""
    input_dir = CoNLL_MULTI_LABEL_NER_DATASETS
    train_snt_num = 5
    gold_datasets = load_gold_datasets(positive_cats, negative_cats, input_dir, train_snt_num)
    assert len(gold_datasets['train']) == train_snt_num

def test_load_gold_medmentions_dataset():
    positive_cats = "T005_T007_T017_T022_T031_T033_T037_T038_T058_T062_T074_T082_T091_T092_T097_T098_T103_T168_T170_T201_T204"
    negative_cats = ""
    input_dir = MEDMENTIONS_MULTI_LABEL_NER_DATASETS
    train_snt_num = 5
    gold_datasets = load_gold_datasets(positive_cats, negative_cats, input_dir, train_snt_num)
    assert len(gold_datasets['train']) == train_snt_num































