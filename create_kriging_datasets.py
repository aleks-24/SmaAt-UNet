import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from utils.interpolation_maps import makeKrigeMap

precipitation_folder = ROOT_DIR / "data" / "precipitation"
#create kriging dataset from regular node dataset
K_SIZE = 256 #size of kriging map

def create_dataset():
    with h5py.File(
        precipitation_folder / "hybrid_train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        train_nodes = orig_f["train"]["nodes"]
        test_nodes = orig_f["test"]["nodes"]

        filename = (
            precipitation_folder / "hybrid_kriging_train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50_size_256.h5"
        )
        #copy images and timestamps, create kriging dataset
        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            f["train"].create_dataset("images", data=train_images, dtype="float32", compression="gzip", compression_opts=9)
            f["train"].create_dataset("timestamps", data=train_timestamps, dtype=h5py.special_dtype(vlen=str), compression="gzip", compression_opts=9)
            f["test"].create_dataset("images", data=test_images, dtype="float32", compression="gzip", compression_opts=9)
            f["test"].create_dataset("timestamps", data=test_timestamps, dtype=h5py.special_dtype(vlen=str), compression="gzip", compression_opts=9)
            train_kriging_dataset = train_set.create_dataset(
                "kriging",
                shape=(5733, 18, 8,K_SIZE,K_SIZE),
                maxshape=(None, 18, 8,K_SIZE,K_SIZE),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_kriging_dataset = test_set.create_dataset(
                "kriging",
                shape=(1557, 18, 8,K_SIZE,K_SIZE),
                maxshape=(None, 18, 8,K_SIZE,K_SIZE),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            #(5733, 22, 18, 8) for train_nodes
            for i in tqdm(range(train_nodes.shape[0])):
                for j in range(train_nodes.shape[2]):
                    input = train_nodes[i][:, j, :]
                    kriging_map = makeKrigeMap(input)
                    train_kriging_dataset[i,j] = kriging_map
            for i in tqdm(range(test_nodes.shape[0])):
                for j in range(test_nodes.shape[2]):
                    input = test_nodes[i][:, j, :]
                    kriging_map = makeKrigeMap(input)
                    test_kriging_dataset[i,j] = kriging_map

create_dataset()