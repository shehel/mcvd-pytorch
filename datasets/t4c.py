import random
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
import h5py
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


import pdb

from clearml import Task

perm = [[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ]
MAX_TEST_SLOT_INDEX = 240
def load_h5_file(file_path: Union[str, Path], sl: Optional[slice] = None, to_torch: bool = False) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.
    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl])
        else:
            data = np.array(data)
        if to_torch:
            data = torch.from_numpy(data)
            data = data.to(dtype=torch.float)
        return data


class T4CDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
            file_filter: str = None,
            im_size = 64
    ):
        """torch dataset from training data.
        Parameters
        ----------
        root_dir
            data root folder, by convention should be `data/raw`, see `data/README.md`. All `**/training/*8ch.h5` will be added to the dataset.
        file_filter: str
            filter files under `root_dir`, defaults to `"**/training/*ch8.h5`
        limit
            truncate dataset size
        transform
            transform applied to both the input and label
        """
        self.root_dir = root_dir
        self.im_size = im_size
        self.files = []
        self.file_filter = file_filter
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"
            #"MOSCOW/training/2019*.h5"#

        self.len = 0
        self.file_list = None
        self.use_npy = False
        self._load_dataset()

    def _load_dataset(self):
        self.file_list = list(Path(self.root_dir).rglob(self.file_filter))

        self.file_list.sort()
        self.len = len(self.file_list) * MAX_TEST_SLOT_INDEX

    def _load_h5_file(self, fn, sl: Optional[slice]):
        if self.use_npy:
            return np.load(fn)
        else:
            return load_h5_file(fn, sl=sl)

    def __len__(self):

        return self.len


    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX
        two_hours = self._load_h5_file(self.file_list[file_idx], sl=slice(start_hour, start_hour + 24))
        #two_hours = self.files[file_idx][start_hour:start_hour+24]

        #input_data, output_data = prepare_test(two_hours)
        input_data = two_hours#, output_data = two_hours[6:12], two_hours[12:18]

        input_data = input_data[:,100:100+self.im_size, 100:100+self.im_size, :]
        #output_data = output_data[:,100:100+self.im_size, 100:100+self.im_size, :]
        # input_data = np.divide(input_data, 255.0) # bring the upper range to 1
        # output_data = np.divide(output_data, 255.0) # bring the upper range to 1
        # input_data = 2*input_data - 1
        # output_data = 2*output_data - 1
        input_data = np.divide(input_data, 255.0) # bring the upper range to 1

        return input_data, input_data[12:]

def train_collate_fn(batch):
    dynamic_input_batch, target_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = np.moveaxis(dynamic_input_batch, source=4, destination=2)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = np.moveaxis(target_batch, source=4, destination=2)
    target_batch = torch.from_numpy(target_batch).float()

    return dynamic_input_batch, target_batch
