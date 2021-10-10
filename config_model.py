from typing import Any

from pydantic import BaseModel


class TrainConfig(BaseModel):
    num_epochs: int = 20
    epoch_length: int = 100
    batch_size: int = 16
    lr: float = 0.001
    wd: float = 0.01


class TaskConfig(BaseModel):
    class_name: str
    train_kwargs: dict[str, Any]
    validation_kwargs: dict[str, Any]


class ModelConfig(BaseModel):
    class_name: str
    kwargs: dict[str, Any]


class ExperimentConfig(BaseModel):
    name: str
    train: TrainConfig = TrainConfig()
    task: TaskConfig
    model: ModelConfig
    repeat: int = 1


class ExperimentConfigs(BaseModel):
    configs: list[ExperimentConfig]
