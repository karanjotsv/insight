import os
from util import *
from model import *

from lightning.pytorch.loggers import WandbLogger

# hugging face; for gated repo
os.environ["HF_TOKEN"] = "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"


if __name__ == "__main__":

    path = "../file/VER/llava-1.5-13b-hf.csv"
    # load dataset
    dataset = read_data(path)

    train_ds = CSM_Dataset(dataset, split="train")
    print(f"\nTRAIN: {train_ds.dataset_length}")

    val_ds = CSM_Dataset(dataset, split="validation")
    print(f"VALIDATION: {val_ds.dataset_length}\n")

    # training config
    config = {
        "max_epochs": 10,
        # "val_check_interval": 0.2,  # times we want to validate during an epoch
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 2,
        # "seed": 98,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./",
        "verbose": True,
    }

    wandb_logger = WandbLogger(project=WANDB_PROJECT, name=WANDB_NAME)
    ##
    model_name = "llava-hf/llava-1.5-13b-hf"
    # init
    model_module = LlaVa(model_name, config, train_ds, val_ds)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        default_root_dir=config.get("result_path"),
        callbacks=[PushToHubCallback()]  # early_stop_callback 
    )

    trainer.fit(model_module)
