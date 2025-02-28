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
from data.datasets import CreateDataset
import xarray as xr
from torch.utils.data import Dataset, DataLoader
from loguru import logger

import torch
import torch.nn.functional as F
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import numpy as np
from tqdm import tqdm
import xarray as xr
import torch
from torch.utils.data import DataLoader
from xbatcher import BatchGenerator

import matplotlib.pyplot as plt
from utils.general import load_config


def count_model_parameters(model):
    """
    Count number of parameters in a Pytorch Model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def z_score_normalization(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform Z-score normalization for each channel of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W].

    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = tensor.mean(dim=[2, 3], keepdim=True)  # Mean along H and W for each channel
    std = tensor.std(dim=[2, 3], keepdim=True)    # Standard deviation along H and W for each channel

    # Prevent division by zero
    std = std + 1e-6

    return (tensor - mean) / std

config = load_config()

start_date = "2020-01-01"
end_date = "2020-10-01"

# hr_data = xr.open_dataset(
#     "/home/ubuntu/project/destine-super-resolution/ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-high-sfc-v0.zarr",
#     engine="zarr",
#     chunks={})
lr_data = xr.open_dataset(
    config["dataset"]["lr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})

hr_data = xr.open_dataset(
    config["dataset"]["hr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})

lr_data = lr_data.astype("float32")
latitude_range = tuple(config["dataset"]["latitude_range"])
longitude_range = tuple(config["dataset"]["longitude_range"])
lr = lr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    time=slice(start_date, end_date))

hr_data = hr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    time=slice(start_date, end_date))



hr_data = hr_data[config['dataset']['data_target']]
lr_data = lr_data[config['dataset']['data_variable']]


latitude_range = tuple(config["dataset"]["latitude_range"])
longitude_range = tuple(config["dataset"]["longitude_range"])
lr_data = lr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    )

hr_data = hr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    )

GPU_DEVICE = 0
device = torch.device("cuda",GPU_DEVICE)
batch_generator_lr = BatchGenerator(lr_data, input_dims={"time": config['training']['batch_size'],
                                                         "latitude":  lr_data.sizes['latitude'],
                                                         "longitude": lr_data.sizes['longitude']})
batch_generator_hr = BatchGenerator(hr_data, input_dims={"time": config['training']['batch_size'],
                                                         "latitude":  hr_data.sizes['latitude'],
                                                         "longitude": hr_data.sizes['longitude']})
from models.swinIR import SwinIR

upscale = 8
window_size = 8

# model = SwinIR(in_chans=3, upscale=upscale,
#                 window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
#                 embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

import models.models as models
# Initialize SRResNet model based on config
model = getattr(models, config["model"]["architecture"])
model = model(
    in_channels = 1,
    large_kernel_size=config["model"]["large_kernel_size"],
    small_kernel_size=config["model"]["small_kernel_size"],
    n_channels=config["model"]["n_channels"],
    n_blocks=config["model"]["n_blocks"],
    scaling_factor=config["model"]["scaling_factor"]
)


logger.info("Number of GPU(s) {}: ".format(torch.cuda.device_count()))
logger.info("GPU(s) in used {}: ".format(GPU_DEVICE))
logger.info("------")
logger.info('===> Building model')
device = torch.device("cuda",GPU_DEVICE)
model = model.to(device)
logger.info("Model on: {}: ".format(next(model.parameters()).device))
nb_parameters = count_model_parameters(model=model)
logger.info("Number of parameters {}: ".format(nb_parameters))
logger.info("------")
logger.info("------")
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.01)

NUM_EPOCHS = 30
BATCH_SIZE = 32
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import torch.nn as nn
criterion = nn.MSELoss()

# Train
epoch_loss = 0

for epoch in range(NUM_EPOCHS):

    epoch_losses = AverageMeter()
    eval_losses = AverageMeter()

    model.train()

    with tqdm(total=(len(batch_generator_lr) - len(batch_generator_lr) % BATCH_SIZE), colour='#3eedc4') as t:
        t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

        # Iterate through both batch generators together
        for batch_lr, batch_hr in zip(batch_generator_lr, batch_generator_hr):

            # Load LR and HR batches into memory
            lr_data = batch_lr.load().to_array().values
            hr_data = batch_hr.load().to_array().values

            # Convert to PyTorch tensors
            lr_tensor = torch.tensor(lr_data, dtype=torch.float32).to(device)
            hr_tensor = torch.tensor(hr_data, dtype=torch.float32).to(device)

            hr_tensor = torch.permute(hr_tensor, (1, 0, 2, 3))
            lr_tensor = torch.permute(lr_tensor, (1, 0, 2, 3))
            hr_tensor = z_score_normalization(hr_tensor)
            lr_tensor = z_score_normalization(lr_tensor)
            pred_tensor = model(lr_tensor)
            optimizer.zero_grad()
            loss = criterion(pred_tensor, hr_tensor)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            epoch_losses.update(loss.item(), len(lr_tensor))
            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(lr_tensor))

        logger.info("Training avg Loss: {:.6f}".format(eval_losses.avg))

    # # Validation
    # model.eval()
    # val_loss = 0

    # with torch.no_grad():  # Disable gradient calculation for validation
    #     with tqdm(total=(len(batch_generator_lr) - len(batch_generator_lr) % BATCH_SIZE), colour='#ff6f61') as t:
    #         t.set_description('Validation')

    #         for batch_lr, batch_hr in zip(batch_generator_lr, batch_generator_hr):
    #             # Load LR and HR batches into memory
    #             lr_data = batch_lr.load().to_array().values
    #             hr_data = batch_hr.load().to_array().values

    #             # Convert to PyTorch tensors
    #             lr_tensor = torch.tensor(lr_data, dtype=torch.float32).to(device)
    #             hr_tensor = torch.tensor(hr_data, dtype=torch.float32).to(device)

    #             # Permute dimensions (num_vars, batch_size, lat, lon)
    #             hr_tensor = torch.permute(hr_tensor, (1, 0, 2, 3))
    #             lr_tensor = torch.permute(lr_tensor, (1, 0, 2, 3))

    #             # Apply normalization
    #             hr_tensor = z_score_normalization(hr_tensor)
    #             lr_tensor = z_score_normalization(lr_tensor)

    #             # Forward pass only
    #             loss = criterion(model(lr_tensor), hr_tensor)
    #             val_loss += loss.item()

    #             eval_losses.update(loss.item(), len(lr_tensor))
    #             t.set_postfix(val_loss='{:.6f}'.format(eval_losses.avg))
    #             t.update(len(lr_tensor))

    # logger.info("Validation avg Loss: {:.6f}".format(eval_losses.avg))
