import time
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import torch
from xbatcher import BatchGenerator
import matplotlib.pyplot as plt
from utils.general import load_config

config = load_config()

data = xr.open_dataset(
    config["dataset"]["hr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})
data_vars = list(data.data_vars)
start_date = "2025-03-01"
end_date = "2025-03-05"
latitude_range = tuple(config["dataset"]["latitude_range"])
longitude_range = tuple(config["dataset"]["longitude_range"])
data = data.sel(time=slice(start_date, end_date))
data = data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                longitude=slice(longitude_range[0],longitude_range[1]))
num_trials = 10  # Number of repetitions
num_vars_list = []
time_avg_list = []
time_std_list = []
size_data = []
for num_vars in range(1, 15):


    data = xr.open_dataset(
    config["dataset"]["hr_zarr_url"],
    engine="zarr", storage_options={"client_kwargs": {"trust_env": "true"}},
    chunks={})
    data_vars = list(data.data_vars)
    start_date = "2025-03-01"
    end_date = "2025-03-05"
    latitude_range = tuple(config["dataset"]["latitude_range"])
    longitude_range = tuple(config["dataset"]["longitude_range"])
    data = data.sel(time=slice(start_date, end_date))
    data = data.sel(latitude=slice(latitude_range[0],latitude_range[1]),
                    longitude=slice(longitude_range[0],longitude_range[1]))

    selected_vars = data_vars[:num_vars]

    hr_data_subset = data[selected_vars]  # Subset dataset
    batch_generator_hr = BatchGenerator(hr_data_subset, input_dims={
        "time": 64,
        "latitude": hr_data_subset.sizes["latitude"],
        "longitude": hr_data_subset.sizes["longitude"]
    })
    print("####")
    print(num_vars)
    print("####")

    times = []

    for _ in range(num_trials):  # Run multiple trials
        start_time = time.time()

        for batch in batch_generator_hr:

            data_batch = batch.load()

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)
        size = data_batch.nbytes / (1024*1024)
        print(size)

        data_batch = 0

    avg_time = np.mean(times)
    std_time = np.std(times)

    num_vars_list.append(num_vars)
    time_avg_list.append(avg_time)
    time_std_list.append(std_time)
    size_data.append(size)
    print(f"Num Vars: {num_vars}, Avg Time: {avg_time:.4f} sec, Std Dev: {std_time:.4f} sec")

# Plot the results with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(num_vars_list, time_avg_list, yerr=time_std_list, fmt='-o', capsize=4, label="Avg Time ± Std Dev")
plt.xlabel("Number of Data Variables")
plt.ylabel("Time to Load (seconds)")
plt.title("Time to Load vs Number of Data Variables")
plt.grid()
plt.legend()
plt.savefig("load_vs_parameters.png")

import pandas as pd

df = pd.DataFrame(data={"climate_variables":num_vars_list,
                        "time_avg":time_avg_list,
                        "time_std":time_std_list,
                        "size_data":size_data})

df["batch"] = 64
df["fps"] = (df['climate_variables'] *  df["batch"] / df["time_avg"])
df['bandwidth_MBps'] = df['size_data'] / df["time_avg"]
df['bandwidth_Mbps'] = df['size_data'] / df["time_avg"] * 8

df['max_bandwidth_Mbps'] = 25000
df["max_fps"] = df['max_bandwidth_Mbps'] * df["fps"] / df['bandwidth_Mbps']
df.to_csv("benchmark_result.csv")