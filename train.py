from pathlib import Path

import enlighten
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_optimizer import RAdam

from config_model import TrainConfig
from models.agents import StatefulModule
from tasks.base import Task


def train(
    train_config: TrainConfig,
    task: Task,
    model: StatefulModule,
    device: torch.device,
    log_dir: Path,
) -> None:
    num_epochs = train_config.num_epochs
    epoch_length = train_config.epoch_length
    batch_size = train_config.batch_size
    lr = train_config.lr
    wd = train_config.wd

    writer = SummaryWriter(log_dir=log_dir)

    reset_mask = torch.ones((batch_size,), dtype=torch.bool, device=device)

    model.to(device)
    model.init_state(batch_size=batch_size)
    optimizer = RAdam(model.parameters(), lr=lr, weight_decay=wd)

    with enlighten.get_manager() as manager:
        epochs_pbar = manager.counter(total=num_epochs, desc="Training progress", unit="epochs")
        for epoch in range(num_epochs):
            iters_pbar = manager.counter(total=epoch_length, desc="Epoch progress", unit="iters", leave=False)
            losses = []
            for _ in range(epoch_length):
                input_sequence, target_sequence, target_sequence_mask = task.get_batch(
                    batch_size=batch_size, device=device
                )
                loss = torch.zeros((), device=device)
                for input, target, target_mask in zip(input_sequence, target_sequence, target_sequence_mask):
                    output = model(input=input)
                    loss = loss + target_mask[:, None] * F.binary_cross_entropy_with_logits(
                        input=output, target=target, reduction="none"
                    )
                loss = (loss.sum(dim=1) / target_sequence_mask.sum(dim=0)).mean(dim=0)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.detach_state()
                model.reset_state(reset_mask=reset_mask)

                iters_pbar.update()
            iters_pbar.close()
            epochs_pbar.update()

            mean_loss = sum(losses) / len(losses)
            save_name = log_dir / f"{epoch:08d}.ckpt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_name,
            )
            writer.add_scalar("Loss/train", mean_loss, epoch)
        epochs_pbar.close()
