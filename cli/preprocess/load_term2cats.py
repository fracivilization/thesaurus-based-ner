import hydra
from hydra.utils import get_original_cwd
import os
from src.dataset.term2cat.term2cats import (
    Term2CatsConfig,
    register_term2cat_configs,
    load_term2cats,
    log_term2cats,
)
import pickle
from src.dataset.utils import singularize, pluralize
from tqdm import tqdm

register_term2cat_configs(None)


@hydra.main(config_path="../../conf", config_name="load_term2cats")
def main(config: Term2CatsConfig):
    output_path = os.path.join(get_original_cwd(), config.output)
    term2cats = load_term2cats(config)

    print("load inflected term2cats")
    count_inflected_term2cats = dict()
    for term, cats in tqdm(term2cats.items()):
        singularized_term = singularize(term)
        pluralized_term = pluralize(term)

        expanded_terms = set()
        for new_term in [singularized_term, pluralized_term]:
            if new_term != term:
                expanded_terms.add(pluralized_term)

        count_inflected_term2cats[term] = cats
        for expanded_term in expanded_terms:
            if expanded_term not in term2cats:
                count_inflected_term2cats[expanded_term] = cats

    with open(output_path, "wb") as f:
        pickle.dump(count_inflected_term2cats, f)

    log_term2cats(count_inflected_term2cats)


if __name__ == "__main__":
    main()
