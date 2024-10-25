from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class CreateDataset(Dataset):
    def __init__(self, hr_data, lr_data, hr_mean, hr_std, lr_mean, lr_std):
        """
        Args:
            hr_data (xarray.DataArray): High-resolution data
            lr_data (xarray.DataArray): Low-resolution data
        """
        self.hr_data = hr_data
        self.lr_data = lr_data
        self.hr_mean, self.hr_std = hr_mean, hr_std
        self.lr_mean, self.lr_std = lr_mean, lr_std

        # Define the transformations for high-resolution and low-resolution data
        self.hr_transform = transforms.Normalize(mean=[hr_mean], std=[hr_std])
        self.lr_transform = transforms.Normalize(mean=[lr_mean], std=[lr_std])

    def __len__(self):
        # Dataset length is the number of time steps
        return self.hr_data.shape[0]

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return
        Returns:
            tuple: (low-resolution, high-resolution) data as tensors
        """

        # Select the time slice at the given index
        hr_sample = self.hr_data.isel(time=idx).values  # (lat, lon)
        lr_sample = self.lr_data.isel(time=idx).values  # (lat, lon)
        # Convert to torch tensors and reshape to match (C, H, W) format
        hr_sample = torch.tensor(hr_sample, dtype=torch.float32).unsqueeze(0)
        lr_sample = torch.tensor(lr_sample, dtype=torch.float32).unsqueeze(0)

        hr_sample = self.hr_transform(hr_sample)
        lr_sample = self.lr_transform(lr_sample)

        return lr_sample, hr_sample