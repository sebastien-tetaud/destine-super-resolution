import os
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import lightning as L
from utils.general import get_bbox_from_config


class TrainerSr(L.LightningModule):
    def __init__(self, config, model):
        super().__init__()

        self.bbox = get_bbox_from_config(config=config)
        self.lr = config['training']['learning_rate']
        self.model = model
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        # self.lr_mean = config["dataset"]["lr_mean"]
        # self.lr_std = config["dataset"]["hr_std"]
        # self.hr_mean = config["dataset"]["hr_mean"]
        # self.hr_std = config["dataset"]["lr_std"]
        self.cmap = config['visualization']['color_map']
        self.loss_function = config['training']['loss_function']
        self.opt = config['training']['optimizer']

    def forward(self, x):
        upscaled_img = self.model(x)
        return upscaled_img

    def training_step(self, batch, batch_idx):

        lr_img, hr_img = batch
        sr_img = self(lr_img)
        # loss = nn.functional.mse_loss(sr_img, hr_img)
        loss_function = getattr(nn.functional, self.loss_function)
        loss = loss_function(sr_img, hr_img)

        psnr_value = self.psnr(sr_img, hr_img)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_psnr', psnr_value, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        lr_img, hr_img = batch
        sr_img = self(lr_img)
        loss_function = getattr(nn.functional, self.loss_function)
        loss = loss_function(sr_img, hr_img)
        psnr_value = self.psnr(sr_img, hr_img)
        ssim_value = self.ssim(sr_img, hr_img)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_psnr', psnr_value, prog_bar=True)
        self.log('val_ssim', ssim_value, prog_bar=True)
        # Log images to TensorBoard every 1 steps
        if batch_idx == 1:
            self.log_images(
                lr_img,
                hr_img,
                sr_img,
                self.current_epoch,
                batch_idx)
        return {
            'val_loss': loss,
            'val_psnr': psnr_value,
            'val_ssim': ssim_value}

    def test_step(self, batch):
        lr_img, hr_img = batch
        sr_img = self(lr_img)
        loss_function = getattr(nn.functional, self.loss_function)
        loss = loss_function(sr_img, hr_img)
        psnr_value = self.psnr(sr_img, hr_img)
        ssim_value = self.ssim(sr_img, hr_img)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_psnr', psnr_value, prog_bar=True)
        self.log('test_ssim', ssim_value, prog_bar=True)

        return {
            'test_loss': loss,
            'test_psnr': psnr_value,
            'test_ssim': ssim_value}

    def configure_optimizers(self):

        optimizer = getattr(torch.optim, self.opt)
        optimizer = optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def log_images(self, lr_img, hr_img, sr_img, epoch, batch_idx):
        """
        Helper function to log input, target, and output images to TensorBoard, including cartopy plots
        """
        # Ensure images are in range [0, 1]
        lr_img = lr_img.cpu()
        hr_img = hr_img.cpu()
        sr_img = sr_img.cpu()

        lr_img = (lr_img * self.lr_std) + self.lr_mean
        hr_img = (hr_img * self.hr_std) + self.hr_mean

        # Convert lr_img, hr_img to numpy for cartopy visualization
        lr_img_np = lr_img.squeeze().numpy()[1]  # (lat, lon) format
        hr_img_np = hr_img.squeeze().numpy()[1]  # (lat, lon) format
        sr_img_np = sr_img.squeeze().numpy()[1]  # (lat, lon) format
        sr_img_np = (sr_img_np * self.hr_std) + self.hr_mean
        # Define color limits based on the range of values
        v_min = min(lr_img_np.min(), hr_img_np.min(), sr_img_np.min())
        v_max = max(lr_img_np.max(), hr_img_np.max(), sr_img_np.max())
        # Create a figure with three subplots for LR, HR, and SR images
        fig, ax = plt.subplots(
            1, 3, figsize=(
                18, 6), subplot_kw={
                'projection': ccrs.Mercator()})
        # Plot low-resolution data
        ax[0].coastlines()
        ax[0].add_feature(cf.BORDERS)
        ax[0].set_extent(self.bbox, crs=ccrs.PlateCarree())
        lr_plot = ax[0].imshow(
            lr_img_np,
            origin='upper',
            extent=self.bbox,
            transform=ccrs.PlateCarree(),
            cmap=self.cmap,
            vmin=v_min,
            vmax=v_max)
        ax[0].set_title("Low-Resolution")
        # Plot high-resolution data
        ax[1].coastlines()
        ax[1].add_feature(cf.BORDERS)
        ax[1].set_extent(self.bbox, crs=ccrs.PlateCarree())
        hr_plot = ax[1].imshow(
            hr_img_np,
            origin='upper',
            extent=self.bbox,
            transform=ccrs.PlateCarree(),
            cmap=self.cmap,
            vmin=v_min,
            vmax=v_max)
        ax[1].set_title("High-Resolution")
        # Plot super-resolved data
        ax[2].coastlines()
        ax[2].add_feature(cf.BORDERS)
        ax[2].set_extent(self.bbox, crs=ccrs.PlateCarree())
        sr_plot = ax[2].imshow(
            sr_img_np,
            origin='upper',
            extent=self.bbox,
            transform=ccrs.PlateCarree(),
            cmap=self.cmap,
            vmin=v_min,
            vmax=v_max)
        ax[2].set_title("Super-Resolution Prediction")
        # Add a colorbar outside the last plot (super-resolution)
        cbar = fig.colorbar(sr_plot, ax=ax, location='right')
        cbar.set_label("Temperature (C)")  # Add units to the colorbar
        # Get the current logging directory (e.g., lightning_logs/version_0)
        log_dir = self.logger.log_dir
        save_dir = os.path.join(log_dir, 'val_prediction')
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Save the figure to a PNG file
        plt.savefig(
            f"{save_dir}/batch_{
                str(batch_idx)}_epoch_{
                str(epoch)}.png",
            format='png',
            bbox_inches='tight')
        plt.close(fig)
