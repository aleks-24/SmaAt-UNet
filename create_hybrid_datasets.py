import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def create_dataset(input_length: int, image_ahead: int, rain_amount_thresh: float):
    """Create a dataset that has target images containing at least `rain_amount_thresh` (percent) of rain."""

    precipitation_folder = ROOT_DIR / "data" / "precipitation"
    nodes_folder = Path(ROOT_DIR / "data" / "Set6")
    nodes = [f for f in nodes_folder.iterdir() if f.is_dir()]
    num_nodes = len(nodes)
    scaler = StandardScaler()
    nodes_train = torch.empty(size= (num_nodes, 314354, 8), dtype=torch.float) #157479 ackshually  #Target sizes: [157536, 8].  Tensor sizes: [157344, 8] should be 314958 
    nodes_test = torch.empty(size= (num_nodes, 105011, 8), dtype=torch.float) #should be 105021
    train_scale_values = torch.empty(size= (num_nodes, 2, 8), dtype=torch.float)
    test_scale_values = torch.empty(size= (num_nodes, 2, 8), dtype=torch.float)

    bad_ind = pd.read_csv('bad_index_train.csv')
    bad_ind['DTG'] = pd.to_datetime(bad_ind['DTG']) 
    bad_ind_test = pd.read_csv('bad_index_test.csv')
    bad_ind_test['DTG'] = pd.to_datetime(bad_ind_test['DTG'])

    for i,node in enumerate(nodes):
        df = pd.read_csv(node / "data.csv")
        df['DTG'] = pd.to_datetime(df['DTG']) #166469
        #make node_train have years 
        
        node_train = df[df['DTG'].dt.year < 2019]
        node_train = node_train[node_train['DTG'].dt.year > 2015]
        result = node_train.copy()
        for j, node in node_train.iterrows():
            #print(node)
            date = node['DTG']
            new_date = date + pd.Timedelta(minutes=5)
            result.loc[j, 'DTG'] = new_date
        node_train = pd.concat([node_train, result])
        node_train.sort_values(by='DTG')
        node_train = node_train[~node_train["DTG"].isin(bad_ind['DTG'])]
        
        dates = node_train['DTG']
        node_train = node_train.drop(columns=['DTG'])
        scaled_train = scaler.fit_transform(node_train.values)
        node_train_scaled = pd.DataFrame(scaled_train, columns=node_train.columns)
        train_scale_values[i] = torch.tensor(
                                np.array([scaler.mean_, scaler.var_])
                                ,dtype=torch.float)
        nodes_train[i] = torch.tensor(node_train_scaled.values, dtype=torch.float)

        node_test = df[df['DTG'].dt.year == 2019]
        result = node_test.copy()
        for j, node in node_test.iterrows():
            date = node['DTG']
            new_date = date + pd.Timedelta(minutes=5)
            result.loc[j, 'DTG'] = new_date
        node_test = pd.concat([node_test, result])
        node_test.sort_values(by='DTG')
        node_test = node_test[~node_test["DTG"].isin(bad_ind_test['DTG'])]
        
        dates = node_test['DTG']
        node_test = node_test.drop(columns=['DTG'])
        scaled_test = scaler.fit_transform(node_test.values)
        node_test_scaled = pd.DataFrame(scaled_test, columns=node_test.columns)
        test_scale_values[i] = torch.tensor(
                                np.array([scaler.mean_, scaler.var_])
                                ,dtype=torch.float)
        nodes_test[i] = torch.tensor(node_test_scaled.values, dtype=torch.float)
    
    #save scale values for normalization
    np.save(precipitation_folder / "train_scales.txt", train_scale_values)
    np.save(precipitation_folder / "test_scales.txt", test_scale_values)

    with h5py.File(
        precipitation_folder / "RAD_NL25_RAC_5min_train_test_2016-2019.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        train_nodes = nodes_train
        test_nodes = nodes_test
        print("Train shape", train_images.shape)
        print("Test shape", test_images.shape)
        imgSize = train_images.shape[1]
        num_pixels = imgSize * imgSize

        filename = (
            precipitation_folder / f"hybrid_train_test_2016-2019_input-length_{input_length}_img-"
            f"ahead_{image_ahead}_rain-threshold_{int(rain_amount_thresh * 100)}.h5"
        )

        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            train_timestamp_dataset = train_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
            train_node_dataset = train_set.create_dataset(
                "nodes",
                shape=(1, 22, input_length + image_ahead, 8),
                maxshape=(None, 22, input_length + image_ahead,8),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_image_dataset = test_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgSize, imgSize),
                maxshape=(None, input_length + image_ahead, imgSize, imgSize),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_timestamp_dataset = test_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
            test_node_dataset = test_set.create_dataset(
                "nodes",
                shape=(1, 22, input_length + image_ahead, 8),
                maxshape=(None, 22, input_length + image_ahead, 8),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )

            origin = [[train_images, train_timestamps, train_nodes], [test_images, test_timestamps, test_nodes]]
            datasets = [
                [train_image_dataset, train_timestamp_dataset, train_node_dataset],
                [test_image_dataset, test_timestamp_dataset, test_node_dataset],
            ]
            for origin_id, (images, timestamps, nodes) in enumerate(origin):
                image_dataset, timestamp_dataset, node_dataset = datasets[origin_id]
                first = True
                nullcounter = 0
                for i in tqdm(range(input_length + image_ahead, len(images))):
                    # If threshold of rain is bigger in the target image: add sequence to dataset
                    if (timestamps[i][0] == (b'NA')):
                        nullcounter += 1
                        print(timestamps[i])
                        continue
                    if (np.sum(images[i] > 0) >= num_pixels * rain_amount_thresh):
                        imgs = images[i - (input_length + image_ahead) : i]
                        timestamps_img = timestamps[i - (input_length + image_ahead) : i]
                        node = nodes[:,int((i - (input_length + image_ahead))/2) : int(i/2), :]
                        node = np.repeat(node, 2, axis=1)
                        #                     print(imgs.shape)
                        #                     print(timestamps_img.shape)
                        # extend the dataset by 1 and add the entry
                        if first:
                            first = False
                        else:
                            image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                            timestamp_dataset.resize(timestamp_dataset.shape[0] + 1, axis=0)
                            node_dataset.resize(node_dataset.shape[0] + 1, axis = 0)
                        image_dataset[-1] = imgs
                        timestamp_dataset[-1] = timestamps_img
                        node_dataset[-1] = node
            print(nullcounter)


if __name__ == "__main__":
    print("Creating dataset with at least 20% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.2)
    print("Creating dataset with at least 50% of rain pixel in target image")
    create_dataset(input_length=12, image_ahead=6, rain_amount_thresh=0.5)
