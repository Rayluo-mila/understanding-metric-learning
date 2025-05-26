import logging
import warnings

logging.basicConfig(level=logging.ERROR)
logging.captureWarnings(True)
warnings.filterwarnings("ignore", category=Warning)

import hydra

from pathlib import Path
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="cfgs", config_name="config", version_base=None)
def main(cfg):
    root_dir = Path.cwd()
    if cfg.env_name == "dmc_pixel":
        trainer = DMCPixelTrainer(cfg)
    elif cfg.env_name == "dmc_state":
        trainer = DMCStateTrainer(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        trainer.load_snapshot()
    trainer.train()


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    from trainers.dmc_pixel_trainer import DMCPixelTrainer
    from trainers.dmc_state_trainer import DMCStateTrainer

    main()
