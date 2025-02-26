# Streaming Deep Learning for Super Resolution on DestinE Climate Data

This repository is a demonstrator for using Streaming Deep Learning to perform Super Resolution on DestinE Climate Data. It leverages a ResNet model with 8x super-resolution capability, applied to climate datasets streamed directly from the Earth DataHub service. The repository also provides a web application for visualizing inference results and generating GeoTIFF files from both low-resolution/high-resolution outputs.

## Key Features:

$\textcolor{orange}{\textsf{Model Architecture}}$: ResNet architecture designed for an 8x super-resolution task.
Data Streaming: Climate data are streamed via the [Earth DataHub service](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-high-sfc) into a DataLoader for real-time processing.

$\textcolor{orange}{\textsf{Data}}$:

- [Low Resolution (LR)](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-standard-sfc): Climate Digital Twin (DT) temperature at 2 meters (t2m), IFS-NEMO model, hourly data on single levels.
- [Ground Truth (HR)](https://earthdatahub.destine.eu/collections/d1-climate-dt-ScenarioMIP-SSP3-7.0-IFS-NEMO/datasets/0001-high-sfc): High-Resolution (HR) Climate Digital Twin temperature at 2 meters (t2m), IFS-NEMO model, hourly data on single levels.

$\textcolor{orange}{\textsf{Data Streaming}}$: Climate data are streamed via the Earth DataHub service into a DataLoader for real-time processing without the need to download them locally.


$\textcolor{orange}{\textsf{Web Application}}$: To visualize inference results and generating GeoTIFF files from both low-resolution/high-resolution outputs.


## Prerequisites

1. Install Python
    Download and install Python
    ```bash
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh
    ```

2. Clone the repository:
    ```bash
    git git@github.com:tetaud-sebastien/DestinE_eXchange_SR.git
    cd DestinE_eXchange_SR
    ```

3. Install the required packages
    Create python environment:
    ```bash
    conda create --name env python==3.12
    ```
    Activate the environment

    ```bash
    conda activate env
    ```
    Install python package
    ```Bash
    pip install -r requirements.txt
    ```


Install COG file generation dependencies

- gdal_translate -> [Installation on Linux](https://gdal.org/en/latest/download.html)

## Prject Structure

```bash
    .
    ├── README.md
    ├── app.py
    ├── assets
    │   └── banner.svg
    ├── auth
    │   └── cacheb-authentication.py
    ├── cfg
    │   └── config.yaml
    ├── data
    │   ├── datasets.py
    │   └── loaders.py
    ├── inference.py
    ├── models
    │   └── models.py
    ├── notebook.ipynb
    ├── train.py
    ├── trainer.py
    └── utils
        └── general.py
```

## Destine Service Access Login

Before to train your model, please got on the main Jupyter Notebook and the following cells to setup your destine Credentials:

```Bash
%%capture cap
%run auth/cacheb-authentication.py
```

```Bash
output_1 = cap.stdout.split('}\n')
token = output_1[-1][0:-1]

from pathlib import Path
with open(Path.home() / ".netrc", "a") as fp:
    fp.write(token)
```

## Train model

It is possible to train the model either on the following notebook: **notebook.ipynb**

The training script takes a configuration file as input, which parses the training parameters(TODO).
You can also run the script directly using the following command:
```Bash
python train.py
```

You can access to the Tensboard via the following typing the cli in the root directory:
```Bash
tensorboard --logdir .
```

$\textcolor{orange}{\textsf{config file}}$: The training use a yaml configuration file. Feel free to change parameters.

## Tesorboard

In the project root direectory:

```Bash
tensorboard --logdir .
```

## Test Model with Gradio

```Bash
python app.py --model_path your/model/path.pt --config_path your/training/config.json
```


Example:

```Bash
python app.py --model_path lightning_logs/version_0/checkpoints/best-val-ssim-epoch=49-val_ssim=0.54.pt --config_path lightning_logs/version_0/config.json
```
## Future Work

- Expand to N Parameters: Although the current setup processes only one variable (t2m), the trainer will be extended to handle multiple parameters simultaneously.
- Multiple Sources: The architecture is flexible enough to incorporate data from various sources, allowing for a multi-source approach to super-resolution tasks in climate modeling

## Getting Help

Feel free to ask questions at the following email adress: [sebastien.tetaud@esa.int](sebastien.tetaud@eas.int) or open a ticket.