import hydra, json
from omegaconf import DictConfig, OmegaConf

from hw8.hw8 import my_app

@hydra.main(config_path="conf", config_name="config_hw8")
def my_main(cfg: DictConfig):
    my_app(cfg)


if __name__ == "__main__":
    import os
    my_main()
