import argparse
import time
from pathlib import Path

import enlighten
import torch
import yaml  # type: ignore[import]

from config_model import ExperimentConfigs
from models.agents import GRUAgent, NTMAgent
from tasks.copy import CopyTask
from tasks.repeat_copy import RepeatCopyTask
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

    with enlighten.get_manager() as manager:
        for config in experiment_configs.configs:
            exp_repeats_pbar = manager.counter(total=config.repeat, desc=config.name, unit="experiments")
            for i in range(config.repeat):
                log_dir = args.log_dir / f"{config.name} ({i})"
                train_task = globals()[config.task.class_name](**config.task.train_kwargs)
                validation_task = globals()[config.task.class_name](**config.task.validation_kwargs)
                model = globals()[config.model.class_name](**config.model.kwargs)
                train(
                    train_config=config.train,
                    train_task=train_task,
                    validation_task=validation_task,
                    model=model,
                    device=device,
                    log_dir=log_dir,
                    manager=manager,
                )
                exp_repeats_pbar.update()
            exp_repeats_pbar.close()


if __name__ == "__main__":
    main()
