import argparse
import datetime
import gradio as gr
import xarray as xr
import numpy as np
import subprocess
import models.models as models
# import rioxarray  # Ensure rioxarray is imported for spatial data handling
from inference import SuperResolutionInference
import os
import base64
from utils.general import (load_json_config, ColorMappingGenerator)

# CLI Argument Parsing
parser = argparse.ArgumentParser(
    description="Gradio interface for super-resolution inference")
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to the JSON configuration file")
parser.add_argument("--model_path", type=str, required=True,
                    help="Path to the pre-trained model file")
args = parser.parse_args()


def svg_to_base64(file_path):
    """
    Convert an SVG file to Base64 encoding.

    Args:
        file_path (str): Path to the SVG file.

    Returns:
        str: Base64 encoded string of the SVG file.
    """
    with open(file_path, "rb") as f:
        svg_base64 = base64.b64encode(f.read()).decode("utf-8")
    return svg_base64


class GradioInference:
    """
    Class to handle the inference process for super-resolution and COG file generation
    for both low-resolution and super-resolved temperature data.
    """

    def __init__(self, config_path, model_path):
        """
        Initialize the GradioInference class by loading configuration and setting up
        dataset parameters.

        Args:
            config_path (str): Path to the configuration JSON file.
            model_path (str): Path to the pre-trained model file.
        """
        self.model_path = model_path
        print(self.model_path)
        self.config = load_json_config(config_path)
        print(self.model_path)
        self.scale_factor_latitude = self.config["model"]["scaling_factor"]
        lr_data = xr.open_dataset(
            self.config["dataset"]["lr_zarr_url"],
            engine="zarr",
            storage_options={"client_kwargs": {"trust_env": "true"}},
            chunks={},
        )
        lr_data = lr_data[self.config["dataset"]["data_variable"]]
        lr_data = lr_data.astype("float32") - 273.15
        latitude_range = tuple(self.config["dataset"]["latitude_range"])
        longitude_range = tuple(self.config["dataset"]["longitude_range"])
        self.lr = lr_data.sel(
            latitude=slice(
                latitude_range[0],
                latitude_range[1]),
            longitude=slice(
                longitude_range[0],
                longitude_range[1]))
        self.lr.attrs["units"] = "C"
        self.sr_result = None
        self.lr_image = None
        self.lr_mean = self.config["dataset"]["lr_mean"]
        self.hr_mean = self.config["dataset"]["hr_mean"]
        self.lr_std = self.config["dataset"]["lr_std"]
        self.hr_std = self.config["dataset"]["hr_std"]

    def run_inference_on_date(self, selected_date):
        """
        Perform super-resolution inference for the selected date.

        Args:
            selected_date (datetime): The date to be used for selecting the low-resolution data.

        Returns:
            matplotlib.figure.Figure: Figure containing low-resolution and super-resolved images.
        """
        self.current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
        # Load SRResNet model configuration
        large_kernel_size = self.config["model"]["large_kernel_size"]
        small_kernel_size = self.config["model"]["small_kernel_size"]
        n_channels = self.config["model"]["n_channels"]
        n_blocks = self.config["model"]["n_blocks"]
        scaling_factor = self.config["model"]["scaling_factor"]

        sr_model = getattr(models, self.config["model"]["architecture"])
        sr_model = sr_model(
            large_kernel_size=large_kernel_size,
            small_kernel_size=small_kernel_size,
            n_channels=n_channels,
            n_blocks=n_blocks,
            scaling_factor=scaling_factor)
        sr = SuperResolutionInference(
            model_path=self.model_path,
            model_class=sr_model)
        # Convert the selected date to the appropriate format
        formatted_str = datetime.datetime.fromtimestamp(
            selected_date).strftime('%Y-%m-%dT%H:%M')
        # Currently hardcoded for demonstration
        self.lr_image = self.lr.sel(time=formatted_str, method='nearest')
        # Preprocess and perform inference
        preprocessed_image = sr.preprocess(
            self.lr_image, lr_mean=self.lr_mean, lr_std=self.lr_std)
        sr_result = sr.inference(preprocessed_image)

        # Postprocess the result and visualize
        self.sr_result = sr.postprocessing(
            sr_result, self.hr_mean, self.hr_std)
        fig = sr.visualize(lr_image=self.lr_image, sr_image=self.sr_result,
                           lr_time=self.lr_image.time.values
                           )

        return fig

    def generate_cog_file(self):
        """
        Generate a COG file based on the super-resolved image with its latitude/longitude.

        Returns:
            str: Filepath of the generated COG file or an error message if it fails.
        """
        color_mapping_gen = ColorMappingGenerator(
            lr_image=self.lr_image.values,
            sr_result=self.sr_result,
            num_colors=20)
        # Write the RGB mapping to a text file
        color_mapping_gen.write_rgb_mapping_to_file("color_mapping.txt")

        try:
            if self.lr_image is None or self.sr_result is None:
                raise ValueError(
                    "Run inference before generating the COG file.")

            # Generate latitude and longitude
            latitudes = np.linspace(
                self.lr_image.latitude.min(), self.lr_image.latitude.max(),
                int(self.lr_image.shape[0] * self.scale_factor_latitude)
            )
            longitudes = np.linspace(
                self.lr_image.longitude.min(), self.lr_image.longitude.max(),
                int(self.lr_image.shape[1] * self.scale_factor_latitude)
            )

            # Create xarray Dataset for super-resolved data
            ds = xr.Dataset(
                data_vars={"t2m": (["latitude", "longitude"], self.sr_result)},
                coords={"latitude": latitudes, "longitude": longitudes, "time": self.lr_image.time},
                attrs={"description": "Super-resolved 2-meter temperature"}
            )
            var = ds['t2m']
            # Convert to proper CRS and handle missing values
            var = var.rename({'latitude': 'y', 'longitude': 'x'})
            var.rio.write_crs("EPSG:4326", inplace=True)
            var_filled = var.fillna(-9999)  # Replace NaNs with -9999
            # Save the super-resolved image as a TIF file
            tif_filename = 'tif_filename.tif'
            var_filled.rio.to_raster(tif_filename)
            # Convert to VRT and apply color relief
            vrt_filename = 'vrt_filename.vrt'
            output_vrt_filename = 'output_vrt_filename.vrt'
            subprocess.run(
                f"gdal_translate -of VRT {tif_filename} {vrt_filename}",
                shell=True,
                check=True)
            subprocess.run(
                f"gdaldem color-relief {vrt_filename} color_mapping.txt {output_vrt_filename}",
                shell=True,
                check=True)
            # Convert VRT to COG
            output_cog_filename = f"{self.current_time}_hr_cog_file.tif"
            subprocess.run(
                f"gdal_translate -of COG {output_vrt_filename} {output_cog_filename}",
                shell=True,
                check=True)
            subprocess.run(
                f"rm -fr {tif_filename} {vrt_filename} {output_vrt_filename}",
                shell=True,
                check=True)

            return output_cog_filename  # Return the COG file path to be downloadable

        except subprocess.CalledProcessError as e:
            return f"Error occurred: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

    def generate_lr_image_cog_file(self):
        """
        Generate a COG file based on the low-resolution image and its latitude/longitude.

        Returns:
            str: Filepath of the generated COG file or an error message if it fails.
        """
        try:
            if self.lr_image is None:
                raise ValueError(
                    "Run inference before generating the COG file.")
            # Create xarray Dataset for the low-resolution image
            ds_lr = xr.Dataset(
                data_vars={
                    "t2m": (
                        [
                            "latitude",
                            "longitude"],
                        self.lr_image.values)},
                coords={
                    "latitude": self.lr_image.latitude,
                    "longitude": self.lr_image.longitude,
                    "time": self.lr_image.time},
                attrs={
                    "description": "Low-resolution 2-meter temperature"})
            var_lr = ds_lr['t2m']
            # Convert to proper CRS and handle NaN values
            var_lr = var_lr.rename({'latitude': 'y', 'longitude': 'x'})
            var_lr.rio.write_crs("EPSG:4326", inplace=True)
            var_lr_filled = var_lr.fillna(-9999)  # Replace NaNs with -9999
            # Save the low-resolution data as a TIF file
            tif_lr_filename = 'lr_tif_filename.tif'
            var_lr_filled.rio.to_raster(tif_lr_filename)
            # Convert to VRT and apply color relief (Optional, depends on your
            # needs)
            vrt_lr_filename = 'lr_vrt_filename.vrt'
            output_vrt_lr_filename = 'lr_output_vrt_filename.vrt'
            subprocess.run(
                f"gdal_translate -of VRT {tif_lr_filename} {vrt_lr_filename}",
                shell=True,
                check=True)
            subprocess.run(
                f"gdaldem color-relief {vrt_lr_filename} color_mapping.txt {output_vrt_lr_filename}",
                shell=True,
                check=True)
            # Convert VRT to COG
            output_cog_lr_filename = f"{self.current_time}_lr_cog_file.tif"
            subprocess.run(
                f"gdal_translate -of COG {output_vrt_lr_filename} {output_cog_lr_filename}",
                shell=True,
                check=True)
            subprocess.run(
                f"rm -fr {vrt_lr_filename} {output_vrt_lr_filename} {tif_lr_filename}",
                shell=True,
                check=True)
            return output_cog_lr_filename  # Return the COG file path to be downloadable

        except subprocess.CalledProcessError as e:
            return f"Error occurred while generating COG for low-resolution image: {
                str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"


# Initialize Gradio Inference object
inference = GradioInference(
    config_path=args.config_path,
    model_path=args.model_path)

# Path to the local SVG file (modify this path if it's located in another
# folder)
# Update this with the actual path to your SVG file
local_svg_path = "assets/banner.svg"
# Function to convert the SVG file to Base64


if not os.path.exists(local_svg_path):
    raise FileNotFoundError(f"SVG file not found at {local_svg_path}")

# Convert SVG to Base64
svg_base64 = svg_to_base64(local_svg_path)
# Create the HTML to embed the SVG using the Base64 data
banner_html = f"""
<div style="text-align: center; margin-bottom: 20px;">
    <img src="data:image/svg+xml;base64,{svg_base64}" alt="Banner" style="width: 100%; max-width: 1200px;" />
</div>
"""

# Define the theme for the Gradio Blocks
with gr.Blocks(theme=gr.themes.Default(primary_hue="orange", secondary_hue="gray")) as demo:
    # Add a banner with the embedded SVG
    gr.HTML(value=banner_html)

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            plot_output = gr.Plot()
            # date_picker = Calendar(type="date", label="Select Date", info="Pick a date from the calendar.")
            date_picker = gr.DateTime(label="Select Date and Time")

            # Run Inference button will automatically take the "primary_hue"
            # color from the theme
            run_inference = gr.Button("Run Inference", variant="primary")
            cog_button = gr.Button("Generate Super-Resolution COG file")
            lr_cog_button = gr.Button("Generate Low-Resolution COG file")
            # Add file outputs for downloading
            output_cog = gr.File(label="Download Super-Resolution COG file")
            output_lr_cog = gr.File(label="Download Low-Resolution COG file")

    # Link the input and output components to the GradioInference class
    run_inference.click(
        fn=inference.run_inference_on_date,
        inputs=date_picker,
        outputs=plot_output,
    )

    cog_button.click(
        fn=inference.generate_cog_file,
        inputs=None,
        outputs=output_cog,  # Use gr.File as output
    )

    lr_cog_button.click(
        fn=inference.generate_lr_image_cog_file,
        inputs=None,
        outputs=output_lr_cog,  # Use gr.File as output
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)
