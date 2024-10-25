import xarray as xr


def load_data(config):
    """
    Load low-resolution (LR) and high-resolution (HR) data based on the configuration.

    Args:
        config (dict): Configuration dictionary containing dataset paths and other settings.

    Returns:
        lr (xarray.DataArray): Low-resolution dataset.
        hr (xarray.DataArray): High-resolution dataset.
    """

    lr_data = xr.open_dataset(
        config["dataset"]["lr_zarr_url"],
        engine="zarr",
    storage_options={"client_kwargs": {"trust_env": "true"}},
        chunks={},
    )

    lr_data = lr_data[config["dataset"]["data_variable"]]
    lr_data = lr_data.astype("float32") - 273.15
    latitude_range = tuple(config["dataset"]["latitude_range"])
    longitude_range = tuple(config["dataset"]["longitude_range"])

    lr = lr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]))
    lr = lr.sel(time=config["dataset"]["time_range"]).load()
    # Load high-resolution data
    hr_data = xr.open_dataset(
        config["dataset"]["hr_zarr_url"],
        engine="zarr",
        storage_options={"client_kwargs": {"trust_env": "true"}},
        chunks={},
    )
    hr_data = hr_data[config["dataset"]["data_target"]]
    hr_data = hr_data.astype("float32") - 273.15
    hr = hr_data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]))
    hr = hr.sel(time=config["dataset"]["time_range"])

    hr.attrs["units"] = config["dataset"]["unit"]
    lr.attrs["units"] = config["dataset"]["unit"]
    if config["training"]["streaming"]:
        return lr, hr
    else:
        return lr.load(), hr.load()