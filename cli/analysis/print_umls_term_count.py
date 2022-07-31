from src.dataset.term2cat.terms import load_tui2cui_count
from src.utils.tree_visualize import (
    get_tree_str,
    make_node2count_consistently_with_child2parent,
)
from src.dataset.utils import STchild2parent, tui2ST

if __name__ == "__main__":
    # tui2count = load_tui2count()
    tui2count = load_tui2cui_count()
    ST2count = {tui2ST[tui]: count for tui, count in tui2count.items()}
    ST2count["ROOT"] = 0
    STchild2parent["ROOT"] = 0
    node2count = make_node2count_consistently_with_child2parent(
        STchild2parent, ST2count
    )
    print(get_tree_str(STchild2parent, node2count))

    pass
