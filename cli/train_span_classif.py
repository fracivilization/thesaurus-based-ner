from omegaconf import DictConfig
import hydra


@hydra.main(config_path="../conf", config_name="train_span_classif")
def main(cfg: DictConfig):
    pass


if __name__ == "__main__":
    main()
