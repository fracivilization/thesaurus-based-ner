import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import os
from src.dataset.term2cat.term2cats import (
    Term2CatsConfig,
    register_term2cat_configs,
    load_term2cats_jsonl,
    log_term2cats,
)
import pickle
from src.dataset.utils import singularize, pluralize
from tqdm import tqdm
import tempfile
from pathlib import Path
from src.utils.utils import DoubleArrayDict
import json

register_term2cat_configs(None)


@hydra.main(config_path="../../conf", config_name="load_term2cats")
def main(config: Term2CatsConfig):
    output_path = to_absolute_path(config.output)
    work_dir = tempfile.TemporaryDirectory()
    term2cats_jsonl = load_term2cats_jsonl(config, work_dir=work_dir)
    print("term2cats_jsonl is loaded")
    term2cats = DoubleArrayDict.load_from_jsonl_key_value_pairs(term2cats_jsonl)
    inflected_term2cats_file = output_path + ".src.inflected_term2cats.jsonl"

    print("load inflected term2cats")
    # TODO: term2cats_jsonlの行数を数えてtqdmできるようにする
    term2cats_jsonl_line_count = sum(1 for _ in open(term2cats_jsonl))
    with open(term2cats_jsonl) as f_input:
        with open(inflected_term2cats_file, "w") as f_output:
            for line in tqdm(f_input, total=term2cats_jsonl_line_count):
                line = line.strip()
                if line:
                    term, cats = json.loads(line)
                f_output.write(json.dumps([term, cats]) + "\n")

                singularized_term = singularize(term)
                pluralized_term = pluralize(term)

                expanded_terms = set()
                for new_term in [singularized_term, pluralized_term]:
                    if new_term != term:
                        expanded_terms.add(new_term)

                for expanded_term in expanded_terms:
                    if expanded_term not in term2cats:
                        f_output.write(json.dumps([expanded_term, cats]) + "\n")

    del term2cats
    inflected_term2cats = DoubleArrayDict.load_from_jsonl_key_value_pairs(
        inflected_term2cats_file
    )
    with open(output_path, "wb") as f:
        pickle.dump(inflected_term2cats, f)

    # TODO: ログを取れるようにする
    # log_term2cats(inflected_term2cats)


if __name__ == "__main__":
    main()
