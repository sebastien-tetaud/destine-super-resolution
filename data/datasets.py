import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


def z_score_normalization(tensor):
    """
    Perform Z-score normalization for each channel of the input tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape [C, H, W]

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = tensor.mean(dim=[1, 2], keepdim=True)  # Mean along H and W for each channel
    std = tensor.std(dim=[1, 2], keepdim=True)    # Standard deviation along H and W for each channel
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor  # FIX: Return the normalized tensor


class CreateDataset(Dataset):
    def __init__(self, lr_data, hr_data):
        """
        Args:
            hr_data (xarray.Dataset): High-resolution data
            lr_data (xarray.Dataset): Low-resolution data
        """
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.lr_variables = list(self.lr_data.data_vars)
        self.hr_variables = list(self.hr_data.data_vars)

    def __len__(self):
        # Dataset length is the number of time steps
        return self.hr_data.dims["time"]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return
        Returns:
            tuple: (low-resolution, high-resolution) data as tensors
        """

        # Select the time slice at the given index
        hr_sample = self.hr_data.isel(time=idx)  # Extract HR data at idx
        lr_sample = self.lr_data.isel(time=idx)  # Extract LR data at idx

        # Convert to NumPy arrays (C, H, W)
        hr_sample = hr_sample.to_array().values  # Convert xarray dataset to numpy array
        lr_sample = lr_sample.to_array().values


        # Convert to torch tensors
        hr_sample = torch.tensor(hr_sample, dtype=torch.float32)
        lr_sample = torch.tensor(lr_sample, dtype=torch.float32)

        # Apply Z-score normalization
        hr_sample = z_score_normalization(hr_sample)
        lr_sample = z_score_normalization(lr_sample)

        return lr_sample, hr_sample


# class StreamCreateDataset(Dataset):
#     def __init__(self, hr_data, lr_data, hr_mean, hr_std, lr_mean, lr_std, batch_size):
#         """
#         Args:
#             hr_data (xarray.DataArray): High-resolution data
#             lr_data (xarray.DataArray): Low-resolution data
#             hr_mean, hr_std: mean and std for HR data normalization
#             lr_mean, lr_std: mean and std for LR data normalization
#             batch_size: the size of the batch to download at once
#         """
#         self.hr_data = hr_data
#         self.lr_data = lr_data
#         self.hr_mean, self.hr_std = hr_mean, hr_std
#         self.lr_mean, self.lr_std = lr_mean, lr_std
#         self.batch_size = batch_size

#         # Define the transformations for high-resolution and low-resolution data
#         self.hr_transform = transforms.Normalize(mean=[hr_mean], std=[hr_std])
#         self.lr_transform = transforms.Normalize(mean=[lr_mean], std=[lr_std])

#         # Calculate the number of batches
#         self.num_batches = int(np.ceil(len(self.hr_data.time) / batch_size))

#     def __len__(self):
#         # Dataset length is the number of batches
#         return self.num_batches

#     def __getitem__(self, idx):
#         """
#         Args:
#             idx (int): Index of the batch to return
#         Returns:
#             tuple: (low-resolution, high-resolution) data as tensors
#         """
#         if idx >= self.num_batches:
#             raise IndexError("Index out of bounds")

#         # Select a batch of time slices instead of a single time slice
#         batch_start = idx * self.batch_size
#         batch_end = min(batch_start + self.batch_size, len(self.hr_data.time))

#         # Download a batch of HR and LR data
#         hr_batch = self.hr_data[batch_start:batch_end].load().values
#         lr_batch = self.lr_data[batch_start:batch_end].load().values

#         # Convert to torch tensors and reshape to match (N, H, W) format
#         hr_batch = torch.tensor(hr_batch, dtype=torch.float32).unsqueeze(1)
#         lr_batch = torch.tensor(lr_batch, dtype=torch.float32).unsqueeze(1)

#         # Apply normalization transforms
#         hr_batch = self.hr_transform(hr_batch)
#         lr_batch = self.lr_transform(lr_batch)

#         return lr_batch, hr_batch