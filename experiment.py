import argparse
import time
from pathlib import Path

import torch
import yaml  # type: ignore[import]

from config_model import ExperimentConfigs
from models.agents import GRUAgent, NTMAgent
from tasks.copy import CopyTask
from train import train


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", type=Path, help="Path to experiment config yaml")
    parser.add_argument("--device", type=str, default="cuda", help="PyTorch device, `cpu` or `cuda`")
    parser.add_argument(
        "--log-dir", type=Path, default=Path("output") / time.strftime("%Y%m%d-%H%M%S"), help="Log directory"
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        experiment_configs_dict = yaml.load(file, Loader=yaml.FullLoader)
    experiment_configs = ExperimentConfigs(**experiment_configs_dict)
    device = torch.device(args.device)

    args.log_dir.mkdir(parents=True, exist_ok=False)
    with open(args.log_dir / "experiment_config.yml", "w") as file:
        yaml.dump(experiment_configs.dict(), file)

    for config in experiment_configs.configs:
        task = CopyTask(**config.task.kwargs)
        model = globals()[config.model.class_name](**config.model.kwargs)
        train(train_config=config.train, task=task, model=model, device=device, log_dir=args.log_dir / config.name)


if __name__ == "__main__":
    main()
