from torch.utils.data import Dataset
import h5py
import numpy as np


class precipitation_maps_h5_hybrid(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super().__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images + num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (num_input_images + num_output_images)
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None
        self.node_dataset = None

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "images"
            ]
            self.node_dataset = h5py.File(self.file_ name, "r")["train" if self.train else "test"]["nodes"]
        imgs = np.array(self.dataset[index : index + self.sequence_length], dtype="float32")
        nodes = np.array(self.node_dataset[index : index + self.sequence_length], dtype = "float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]
        input_nodes = nodes[: self.num_input]
        target_node = nodes[-1]

        return input_img, target_img, input_nodes, target_node

    def __len__(self):
        return self.size_dataset

class precipitation_maps_h5_kriging(Dataset):
    def __init__(self, in_file, num_input_images, num_output_images, train=True, transform=None):
        super().__init__()

        self.file_name = in_file
        self.n_images, self.nx, self.ny = h5py.File(self.file_name, "r")["train" if train else "test"]["images"].shape

        self.num_input = num_input_images
        self.num_output = num_output_images
        self.sequence_length = num_input_images + num_output_images

        self.train = train
        # Dataset is all the images
        self.size_dataset = self.n_images - (num_input_images + num_output_images)
        # self.size_dataset = int(self.n_images/(num_input_images+num_output_images))
        self.transform = transform
        self.dataset = None
        self.node_dataset = None

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_name, "r", rdcc_nbytes=1024**3)["train" if self.train else "test"][
                "images"
            ]
            self.kriging_dataset = h5py.File(self.file_ name, "r")["train" if self.train else "test"]["kriging"]
        imgs = np.array(self.dataset[index : index + self.sequence_length], dtype="float32")
        kriging_maps = np.array(self.kriging_dataset[index : index + self.sequence_length], dtype = "float32")

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)
        input_img = imgs[: self.num_input]
        target_img = imgs[-1]
        input_kriging_maps = kriging_maps[: self.num_input]
        target_kriging_map = kriging_maps[-1]

        return input_img, target_img, input_kriging_maps, target_kriging_map

    def __len__(self):
        return self.size_dataset
