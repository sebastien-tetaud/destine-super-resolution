import warnings
import torch
import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.general import (get_bbox_from_config,
                           load_config, compute_mean_std,
                           create_gif_from_images,
                           save_best_model_as_pt,
                           save_config_to_log_dir)
from data.loaders import load_data
from data.datasets import CreateDataset, StreamCreateDataset
import models.models as models
from trainer import TrainerSr
# Ignore warnings and set precision
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')


def main():
    """
    Main function to load data, initialize model, and train using PyTorch Lightning.
    """
    # Load configuration
    config = load_config()

    seed_everything(config["training"]["seed"], workers=True)

    # Load bounding box and data (low-res and high-res)
    bbox = get_bbox_from_config(config=config)
    lr, hr = load_data(config=config)

    # Compute mean and standard deviation for normalization
    hr_mean, hr_std = compute_mean_std(hr)
    lr_mean, lr_std = compute_mean_std(lr)

    config["dataset"]["hr_mean"] = hr_mean
    config["dataset"]["hr_std"] = hr_std
    config["dataset"]["lr_mean"] = lr_mean
    config["dataset"]["lr_std"] = lr_std
    # Create DataLoader for train, validation, and test sets
    batch_size = config['training']['batch_size']

    # Split data into train, validation, and test indices
    time_indices = np.arange(len(hr.time.values))
    train_indices, remaining_indices = train_test_split(
        time_indices, train_size=0.70, shuffle=True,
        random_state=config["training"]["seed"]
    )
    val_indices, test_indices = train_test_split(
        remaining_indices, test_size=config["validation"]["val_split_ratio"], shuffle=False,
        random_state=config["training"]["seed"]
    )

    # Split datasets into train, validation, and test sets
    train_lr, train_hr = lr.isel(
        time=train_indices), hr.isel(
        time=train_indices)
    val_lr, val_hr = lr.isel(time=val_indices), hr.isel(time=val_indices)
    test_lr, test_hr = lr.isel(time=test_indices), hr.isel(time=test_indices)

    # Print dataset size information
    print(f"Train samples: {len(train_hr.time.values)}")
    print(f"Validation samples: {len(val_hr.time.values)}")
    print(f"Test samples: {len(test_hr.time.values)}")

    if config["training"]["streaming"]:

        train_loader = StreamCreateDataset(hr_data=train_hr, lr_data=train_lr,
                                           hr_mean=hr_mean, hr_std=hr_std,
                                           lr_mean=lr_mean, lr_std=lr_std,
                                           batch_size=batch_size)

        val_loader = StreamCreateDataset(hr_data=val_hr, lr_data=val_lr,
                                         hr_mean=hr_mean, hr_std=hr_std,
                                         lr_mean=lr_mean, lr_std=lr_std,
                                         batch_size=batch_size)

        # Create test dataset
        test_loader = StreamCreateDataset(hr_data=test_hr, lr_data=test_lr,
                                          hr_mean=hr_mean, hr_std=hr_std,
                                          lr_mean=lr_mean, lr_std=lr_std,
                                          batch_size=batch_size)

    else:

        # Create datasets for train, validation, and test
        train_dataset = CreateDataset(
            hr_data=train_hr, lr_data=train_lr, hr_mean=hr_mean, hr_std=hr_std,
            lr_mean=lr_mean, lr_std=lr_std
        )
        val_dataset = CreateDataset(
            hr_data=val_hr, lr_data=val_lr, hr_mean=hr_mean, hr_std=hr_std,
            lr_mean=lr_mean, lr_std=lr_std
        )
        test_dataset = CreateDataset(
            hr_data=test_hr, lr_data=test_lr, hr_mean=hr_mean, hr_std=hr_std,
            lr_mean=lr_mean, lr_std=lr_std
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=14)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=14)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=14)

    # Load SRResNet model configuration
    large_kernel_size = config["model"]["large_kernel_size"]
    small_kernel_size = config["model"]["small_kernel_size"]
    n_channels = config["model"]["n_channels"]
    n_blocks = config["model"]["n_blocks"]
    scaling_factor = config["model"]["scaling_factor"]

    sr_model = getattr(models, config["model"]["architecture"])
    sr_model = sr_model(
        large_kernel_size=large_kernel_size,
        small_kernel_size=small_kernel_size,
        n_channels=n_channels,
        n_blocks=n_blocks,
        scaling_factor=scaling_factor)
    # Initialize the training model with trainer-specific parameters
    model = TrainerSr(config=config, model=sr_model)

    # Define model checkpoint callback
    checkpoint_val_ssim = ModelCheckpoint(
        monitor=config['checkpoint']['monitor'],
        filename="best-val-ssim-{epoch:02d}-{val_ssim:.2f}",
        save_top_k=1,
        mode=config['checkpoint']['mode'])

    # Set up the PyTorch Lightning Trainer
    trainer = L.Trainer(
        devices=config["training"]["devices"],  # Specify GPU devices
        max_epochs=config["training"]["epochs"],
        accelerator=config["training"]["accelerator"],
        deterministic=config["training"]["deterministic"],
        callbacks=[checkpoint_val_ssim],
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Load best model for testing
    best_model_path = checkpoint_val_ssim.best_model_path
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["state_dict"])

    # Test the model
    trainer.test(model, test_loader)

    # Generate visualizations or create a GIF from validation predictions
    create_gif_from_images(trainer=trainer)

    _ = save_best_model_as_pt(checkpoint_val_ssim, sr_model)
    save_config_to_log_dir(log_dir_path=trainer.log_dir, config=config)


if __name__ == "__main__":
    main()
