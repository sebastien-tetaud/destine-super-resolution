import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cf
from torchvision import transforms
from loguru import logger


class SuperResolutionInference:
    def __init__(self, model_path: str, model_class: torch.nn.Module, device: str = 'cuda'):
        """
        Initialize the Super-Resolution inference class.

        Args:
            model_path (str): Path to the trained model checkpoint.
            model_class (torch.nn.Module): Model architecture.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path, model_class)
        logger.info(f"Model loaded and moved to device: {self.device}")

    def load_model(self, model_path: str, model_class: torch.nn.Module) -> torch.nn.Module:
        """
        Load the model from the checkpoint.

        Args:
            model_path (str): Path to the model checkpoint.
            model_class (torch.nn.Module): Model class to initialize.

        Returns:
            model: Loaded model ready for inference.
        """
        logger.info(f"Loading checkpoint from {model_path}")
        model = model_class
        checkpoint = torch.load(model_path, map_location="cpu")

        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])  # For Lightning checkpoint
        else:
            model.load_state_dict(checkpoint)  # Standard checkpoint

        model.eval()
        return model.to(self.device)

    def preprocess(self, lr_data: xr.DataArray, lr_mean: float, lr_std: float) -> torch.Tensor:
        """
        Preprocess low-resolution (LR) data for inference.

        Args:
            lr_data (xarray.DataArray): Low-resolution data.
            lr_mean (float): Mean value for normalization.
            lr_std (float): Standard deviation for normalization.

        Returns:
            torch.Tensor: Preprocessed LR data.
        """
        logger.info(f"Preprocessing data with mean={lr_mean}, std={lr_std}")
        lr_transform = transforms.Normalize(mean=[lr_mean], std=[lr_std])
        lr_tensor = torch.tensor(lr_data.values, dtype=torch.float32).unsqueeze(0)
        lr_tensor = lr_transform(lr_tensor).unsqueeze(0)
        return lr_tensor

    def inference(self, lr_image: torch.Tensor) -> np.ndarray:
        """
        Perform inference using the super-resolution model.

        Args:
            lr_image (torch.Tensor): Preprocessed LR image tensor.

        Returns:
            np.ndarray: Super-resolved image.
        """
        lr_image = lr_image.to(self.device)
        with torch.no_grad():
            sr_image = self.model(lr_image)
        return sr_image.cpu().squeeze().numpy()

    def postprocessing(self, sr_image: np.ndarray, hr_mean: float, hr_std: float) -> np.ndarray:
        """
        Post-process the super-resolved image (denormalization).

        Args:
            sr_image (np.ndarray): Super-resolved image.
            hr_mean (float): Mean value for denormalization.
            hr_std (float): Standard deviation for denormalization.

        Returns:
            np.ndarray: Post-processed image.
        """
        return (sr_image * hr_std) + hr_mean

    def visualize(self, lr_image: xr.DataArray, sr_image: np.ndarray, lr_time: str,
                  lon_min: float = 5, lon_max: float = 16.25, lat_min: float = 47, lat_max: float = 58.22):
        """
        Visualize the LR input and SR output images.

        Args:
            lr_image (xarray.DataArray): Low-resolution image for comparison.
            sr_image (np.ndarray): Super-resolved output image.
            lr_time (str): Timestamp for the LR image.
            lon_min, lon_max, lat_min, lat_max (float): Plot extent coordinates.
        """
        v_min = min(lr_image.values.min(), sr_image.min())
        v_max = max(lr_image.values.max(), sr_image.max())

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.Mercator()})

        # Plot Low-Resolution Image
        ax[0].coastlines()
        ax[0].add_feature(cf.BORDERS)
        ax[0].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        ax[0].imshow(lr_image.values, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                     transform=ccrs.PlateCarree(), cmap="YlOrRd", vmin=v_min, vmax=v_max)
        ax[0].set_title(f"Low-Resolution: {lr_time.astype('datetime64[m]')}")

        # Plot Super-Resolution Image
        ax[1].coastlines()
        ax[1].add_feature(cf.BORDERS)
        ax[1].set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        hr = ax[1].imshow(sr_image, origin='upper', extent=[lon_min, lon_max, lat_min, lat_max],
                          transform=ccrs.PlateCarree(), cmap="YlOrRd", vmin=v_min, vmax=v_max)
        ax[1].set_title(f"Super-Resolution: {lr_time.astype('datetime64[m]')}")

        cbar_hr = fig.colorbar(hr, ax=ax[1])
        cbar_hr.set_label(f"Temperature ({lr_image.units})")

        plt.tight_layout()
        plt.show()

        return fig