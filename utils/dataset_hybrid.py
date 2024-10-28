from torch.utils.data import Dataset
import h5py
import numpy as np


class precipitation_maps_h5_nodes(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super().__init__()

        self.file_name = in_file
        self.samples, _ ,_ ,_ = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train

        self.transform = transform
        self.dataset = None
        self.dataset_node = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset_img = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "images"
            ]
            self.dataset_node = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "nodes"
            ]
        imgs = np.array(self.dataset_img[index], dtype="float32")
        nodes = np.array(self.dataset_node[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        input_nodes = nodes[: self.num_input]
        target_img = imgs[-1]
        target_nodes = nodes[-1]

        return input_img, input_nodes, target_img, target_nodes

    def __len__(self):
        return self.samples

class precipitation_maps_h5_kriging(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super().__init__()

        self.file_name = in_file
        self.samples, _ ,_ ,_ = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images

        self.train = train

        self.transform = transform
        self.dataset = None
        self.dataset_node = None

    def __getitem__(self, index):
        # min_feature_range = 0.0
        # max_feature_range = 1.0
        # with h5py.File(self.file_name, 'r') as dataFile:
        #     dataset = dataFile["train" if self.train else "test"]['images'][index:index+self.sequence_length]
        # load the file here (load as singleton)
        if self.dataset is None:
            self.dataset_img = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "images"
            ]
            self.dataset_node = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "kriging"
            ]
        imgs = np.array(self.dataset_img[index], dtype="float32")
        kriging = np.array(self.dataset_node[index], dtype="float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        input_kriging = kriging[: self.num_input]
        target_img = imgs[-1]
        target_kriging = kriging[-1]

        return input_img, input_kriging, target_img, target_kriging

    def __len__(self):
        return self.samples
