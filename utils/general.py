import os
import imageio
from natsort import natsorted
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

import numpy as np
import torch
import os
import yaml
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import json


def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def load_json_config(config_path="lightning_logs/version_0/config.json"):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config



def save_config_to_log_dir(log_dir_path, config):
    """
    Save configuration dictionary as a YAML file in the specified log directory.

    Parameters:
    - log_dir_path (str): Path to the trainer's log directory where config will be saved.
    - config (dict): Configuration dictionary to save.

    Raises:
    - FileNotFoundError: If the specified log directory does not exist.
    """
    # Ensure the log directory exists
    if not os.path.exists(log_dir_path):
        raise FileNotFoundError(f"Log directory '{log_dir_path}' does not exist.")

    # Define the path to the config file within the log directory
    config_path = os.path.join(log_dir_path, "config.json")

    with open(config_path, "w") as outfile:
        json.dump(config, outfile)

        print(f"Configuration file saved to: {config_path}")


def compute_mean_std(data):
    """
    Computes the mean and standard deviation of the entire xarray dataset.
    Args:
        data (xarray.DataArray): Dataset (either HR or LR) to compute stats for
    Returns:
        mean (float), std (float): The mean and standard deviation of the dataset
    """
    mean = data.mean().values
    std = data.std().values

    return np.float64(mean), np.float64(std)


def get_bbox_from_config(config):
    """
    Extracts latitude and longitude ranges from the config dictionary and returns the bounding box.

    Args:
        config (dict): Configuration dictionary containing latitude and longitude ranges.
                       Expected structure:
                       {
                           "dataset": {
                               "longitude_range": [min_longitude, max_longitude],
                               "latitude_range": [min_latitude, max_latitude]
                           }
                       }

    Returns:
        list: Bounding box coordinates as [lon_min, lon_max, lat_min, lat_max].
    """
    lon_min, lon_max = config['dataset']['longitude_range']
    lat_min, lat_max = config['dataset']['latitude_range']

    return [lon_min, lon_max, lat_min, lat_max]


def save_best_model_as_pt(checkpoint_callback, model_instance):
    """
    Saves the best checkpoint as a standard PyTorch .pth file without using Lightning.

    Args:
        checkpoint_callback (ModelCheckpoint): The checkpoint callback used during training.
        model_instance (nn.Module): The PyTorch model instance (the architecture you want to load).
    """
    # Get the path to the best checkpoint (file path)
    best_checkpoint_path = checkpoint_callback.best_model_path
    print(f"Best checkpoint path: {best_checkpoint_path}")

    # Load the checkpoint file (using PyTorch directly)
    checkpoint = torch.load(best_checkpoint_path, map_location=torch.device('cpu'))  # Adjust map_location for GPU if necessary

    # Extract the model's state_dict from the checkpoint
    model_state_dict = checkpoint['state_dict']

    # You might need to adjust key names if they contain "module." or "model."
    model_instance.load_state_dict({k.replace("model.", ""): v for k, v in model_state_dict.items()})

    # Save the PyTorch model's state_dict as .pth
    pth_file_path = best_checkpoint_path.replace("ckpt", "pt")
    torch.save(model_instance.state_dict(), pth_file_path)
    print(f"Best model saved as: {pth_file_path}")
    return pth_file_path


def create_gif_from_images(trainer, output_name="training_progress.gif", duration=10):
    """
    Creates a GIF from PNG images saved in the current trainer log directory.

    Args:
        trainer (pl.Trainer): The Lightning Trainer object to get the log directory.
        output_name (str): The name of the output GIF file.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    log_dir = trainer.log_dir  # Get the current logging directory
    log_dir = os.path.join(log_dir, "val_prediction")

    # Find all PNG files in the log directory
    images = [img for img in os.listdir(log_dir) if img.endswith(".png")]

    # Sort images by name (to ensure they are in correct order)
    images = natsorted(images)


    # Create full paths to images
    image_paths = [os.path.join(log_dir, img) for img in images]

    # Read and create GIF
    gif_path = os.path.join(log_dir, output_name)
    with imageio.get_writer(gif_path, mode='I', duration=duration) as writer:
        for filename in image_paths:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"GIF saved at {gif_path}")


class ColorMappingGenerator:
    def __init__(self, lr_image, sr_result, num_colors=20, colormap="YlOrRd"):
        """
        Initialize the ColorMappingGenerator with image data and parameters.

        Parameters:
        - lr_image: Low-resolution image array (temperature values or similar)
        - sr_result: Super-resolution result array
        - num_colors: Number of discrete colors in the colormap
        - colormap: The colormap to use (default: 'YlOrRd')
        """
        self.num_colors = num_colors
        self.v_min = min(lr_image.min(), sr_result.min())  # Minimum value from both images
        self.v_max = max(lr_image.max(), sr_result.max())  # Maximum value from both images
        self.color_values = np.linspace(self.v_min, self.v_max, self.num_colors)  # Generate evenly spaced values
        self.cmap = plt.get_cmap(colormap)  # Get colormap
        self.norm = mpl.colors.Normalize(vmin=self.v_min, vmax=self.v_max)  # Normalize values
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)  # Colormap scalar

    def write_rgb_mapping_to_file(self, filename="color_mapping.txt"):
        """
        Write the RGB mapping to a text file.

        Parameters:
        - filename: The file to save the RGB mapping (default: 'color_mapping.txt')
        """
        with open(filename, 'w') as file:
            # Add special case for missing values (-9999) as black
            file.write(f"{-9999.0}, {0}, {0}, {0}\n")
            # Write RGB mappings for each value
            for value in self.color_values:
                rgba = self.scalarMap.to_rgba(value)  # Get RGBA color
                rgb = tuple(int(x * 255) for x in rgba[:3])  # Convert to 0-255 range (ignore alpha)
                file.write(f"{value:.2f} {rgb[0]}, {rgb[1]}, {rgb[2]}\n")
        print(f"Color mapping saved to {filename}.")

    def visualize_colorbar(self):
        """
        Visualize the colorbar corresponding to the colormap and value range.
        """
        fig, ax = plt.subplots(figsize=(6, 1))
        fig.subplots_adjust(bottom=0.5)
        # Create and display the colorbar
        colorbar = mpl.colorbar.ColorbarBase(ax, cmap=self.cmap, norm=self.norm, orientation='horizontal')
        colorbar.set_label('Value')
        plt.show()