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


def count_model_parameters(model):
    """
    Count number of parameters in a Pytorch Model.

    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Load configuration and set seed
config = load_config()
latitude_range = tuple(config["dataset"]["latitude_range"])
longitude_range = tuple(config["dataset"]["longitude_range"])
start_date = "2020-01-01"
end_date = "2020-02-01"

# Load
lr_data = xr.open_dataset(
    config["dataset"]["lr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})
lr_data = lr_data.astype("float32")


hr_data = xr.open_dataset(
    config["dataset"]["hr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})
hr_data = hr_data.astype("float32")

hr = hr_data[config['dataset']['data_target']]
lr = lr_data[config['dataset']['data_variable']]

lr = lr.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    time=slice(start_date, end_date))
hr = hr.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]),
                    time=slice(start_date, end_date))



lr_size_in_mb = lr.nbytes / (1024 * 1024)
hr_size_in_mb = hr.nbytes / (1024 * 1024)
print(f"Dataset LR size: {lr_size_in_mb:.2f} MB")
print(f"Dataset LR size: {hr_size_in_mb:.2f} MB")

## Data Access ##

# hr_path = "ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-high-sfc-v0.zarr"
# lr_path = "ScenarioMIP-SSP3-7.0-IFS-NEMO-0001-standard-sfc-v0.zarr"

# lr_loaded = xr.open_dataset(lr_path, engine="zarr")
# hr_loaded = xr.open_dataset(hr_path, engine="zarr")

# Split indices for train, validation, and test sets
batch_size = config['training']['batch_size']
time_indices = np.arange(len(hr.time.values))

train_indices, remaining_indices = train_test_split(
    time_indices, train_size=0.70, shuffle=True, random_state=config["training"]["seed"]
)
val_indices, test_indices = train_test_split(
    remaining_indices, test_size=config["validation"]["val_split_ratio"], shuffle=False, random_state=config["training"]["seed"]
)

# Split indices for train, validation, and test sets
batch_size = config['training']['batch_size']

time_indices = np.arange(len(hr.time.values))
train_indices, remaining_indices = train_test_split(
    time_indices, train_size=0.70, shuffle=True, random_state=config["training"]["seed"]
)
val_indices, test_indices = train_test_split(
    remaining_indices, test_size=config["validation"]["val_split_ratio"], shuffle=False, random_state=config["training"]["seed"]
)


# Split datasets into train, validation, and test sets
train_lr, train_hr = lr.isel(time=train_indices), hr.isel(time=train_indices)
val_lr, val_hr = lr.isel(time=val_indices), hr.isel(time=val_indices)
test_lr, test_hr = lr.isel(time=test_indices), hr.isel(time=test_indices)

# Print dataset sizes
print(f"Train samples: {len(train_hr.time.values)}")
print(f"Validation samples: {len(val_hr.time.values)}")
print(f"Test samples: {len(test_hr.time.values)}")

def normalize_xarray(dataset, variables):
    """
    Normalize an xarray dataset feature-wise for each time index.

    Args:
        dataset (xr.Dataset): Input xarray dataset
        variables (list): List of variable names to normalize

    Returns:
        xr.Dataset: Normalized dataset
    """
    normalized_data = {}

    for v in variables:
        # Compute mean & std for each time step independently
        mean = dataset[v].mean(dim=["latitude", "longitude"])
        std = dataset[v].std(dim=["latitude", "longitude"])

        # Normalize feature-wise per time step
        normalized_data[v] = (dataset[v] - mean) / std

    # Return as a new xarray dataset
    return xr.Dataset(normalized_data, coords=dataset.coords)

def z_score_normalization(tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform Z-score normalization for each channel of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W].

    Returns:
        torch.Tensor: Normalized tensor.
    """
    mean = tensor.mean(dim=[1, 2], keepdim=True)  # Mean along H and W for each channel
    std = tensor.std(dim=[1, 2], keepdim=True)    # Standard deviation along H and W for each channel

    # Prevent division by zero
    std = std + 1e-6

    return (tensor - mean) / std

class CreateDataset(Dataset):
    """
    PyTorch Dataset for handling low-resolution (LR) and high-resolution (HR)
    climate data for super-resolution tasks.

    Attributes:
        hr_data (xarray.Dataset): High-resolution dataset.
        lr_data (xarray.Dataset): Low-resolution dataset.
        hr_variables (list): List of HR variable names.
        lr_variables (list): List of LR variable names.
    """

    def __init__(self, lr_data, hr_data, batch_size):
        """
        Initialize the dataset with low-resolution and high-resolution climate data.

        Args:
            lr_data (xarray.Dataset): Low-resolution dataset.
            hr_data (xarray.Dataset): High-resolution dataset.
        """

        self.hr_data = hr_data
        self.lr_data = lr_data
        self.lr_variables = list(self.lr_data.data_vars)
        self.hr_variables = list(self.hr_data.data_vars)
        self.batch_size = batch_size
        self.num_batches = int(np.ceil(len(self.hr_data.time) / batch_size))

    def __len__(self) -> int:
        """
        Get the total number of time steps in the dataset.

        Returns:
            int: The number of time steps.
        """
        return self.num_batches

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the batch to return
        Returns:
            tuple: (low-resolution, high-resolution) data as tensors
        """
        if idx >= self.num_batches:
            raise IndexError("Index out of bounds")

        # Select a batch of time slices instead of a single time slice
        batch_start = idx * self.batch_size
        batch_end = min(batch_start + self.batch_size, len(self.hr_data.time))



        # Download a batch of HR and LR data
        hr_batch = self.hr_data.sel(time=slice(batch_start,batch_end)).load().values
        lr_batch = self.lr_data.sel(time=slice(batch_start,batch_end)).load().values
        print(hr_batch.shape)
        # Convert to torch tensors and reshape to match (N, H, W) format
        hr_batch = torch.tensor(hr_batch, dtype=torch.float32).unsqueeze(1)
        lr_batch = torch.tensor(lr_batch, dtype=torch.float32).unsqueeze(1)


        return lr_batch, hr_batch

## Feature-wise Normalization
input_variables = list(train_lr.data_vars)
output_variables = list(train_hr.data_vars)
print(f"Inputs samples: {input_variables}")
print(f"Target samples: {output_variables}")


train_dataset = CreateDataset(lr_data=train_lr, hr_data=train_hr, batch_size=batch_size)
val_dataset = CreateDataset(lr_data=train_lr, hr_data=val_hr, batch_size=batch_size)
test_dataset = CreateDataset(lr_data=test_lr, hr_data=test_hr, batch_size=batch_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)


from models.swinIR import SwinIR


upscale = 8
window_size = 8

model = SwinIR(in_chans=3, upscale=upscale,
                window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')

GPU_DEVICE = 0

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

NUM_EPOCHS = 10
BATCH_SIZE = 16
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

    with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE), colour='#3eedc4') as t:
        t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

        for data in train_loader:

            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            epoch_losses.update(loss.item(), len(inputs))
            t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
            t.update(len(inputs))
        logger.info("Training avg Loss: {:.6f}".format(eval_losses.avg))

    # Validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        with tqdm(total=(len(val_dataset) - len(val_dataset) % BATCH_SIZE), colour='#f542a4') as t:
            t.set_description('Validation')

            for data in val_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                eval_losses.update(loss.item(), len(inputs))
                t.set_postfix(loss='{:.6f}'.format(eval_losses.avg))
                t.update(len(inputs))

    logger.info("Validation avg Loss: {:.6f}".format(eval_losses.avg))

