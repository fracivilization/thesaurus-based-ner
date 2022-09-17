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

register_term2cat_configs(None)


@hydra.main(config_path="../../conf", config_name="load_term2cats")
def main(config: Term2CatsConfig):
    output_path = os.path.join(get_original_cwd(), config.output)
    term2cats = load_term2cats(config)
    with open(output_path, "wb") as f:
        pickle.dump(term2cats, f)

    with open(output_path, "rb") as f:
        term2cats = pickle.load(f)
    log_term2cats(term2cats)


if __name__ == "__main__":
    main()
