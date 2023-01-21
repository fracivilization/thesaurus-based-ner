from src.dataset.utils import get_umls_negative_cats_from_focus_cats


def test_get_umls_negative_cats_from_focus_cats():
    focus_cats = ["T005"]
    negative_cats = get_umls_negative_cats_from_focus_cats(focus_cats)
    assert [
        "T007",
        "T017",
        "T051",
        "T073",
        "T077",
        "T167",
        "T194",
        "T204",
    ] == negative_cats
