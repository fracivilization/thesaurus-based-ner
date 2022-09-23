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
    
    print('load inflected term2cats')
    count_inflected_term2cats = dict()
    for term, cats in tqdm(term2cats.items()):
        sigularized_term = singularize(term)
        pluralized_term = pluralize(term)
        count_inflected_term2cats[sigularized_term] = cats
        count_inflected_term2cats[pluralized_term] = cats

    with open(output_path, "wb") as f:
        pickle.dump(count_inflected_term2cats, f)

    log_term2cats(count_inflected_term2cats)


if __name__ == "__main__":
    main()
