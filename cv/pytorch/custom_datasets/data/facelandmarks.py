import torch
import pytorch_lightning as pl
import os
import numpy as np
import pandas as pd
from custom_datasets.data.base_data_module import BaseDataModule
from custom_datasets.data.base_data_module import _download_raw_dataset
from torch.utils.data import random_split
from custom_datasets.data.util import BaseDataset
import toml
import zipfile
from pathlib import Path
import argparse
import cv2
from torchvision import transforms


DL_DATA_DIRNAME = BaseDataModule.data_dirname()
METADATA_FILENAME = DL_DATA_DIRNAME / "metadata.toml"
METADATA = toml.load(METADATA_FILENAME)


class FaceLandmarksDataset(BaseDataModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize(size=(224, 224))
                                             ])

    def prepare_data(self):
        _download_raw_dataset(METADATA, DL_DATA_DIRNAME)

    def setup(self):
        img_sequence, target_sequence = _process_raw_data(METADATA["filename"],
                                                          DL_DATA_DIRNAME)
        full_ds = BaseDataset(img_sequence, target_sequence,
                              transform=self.transform)
        self.data_train, self.data_val, self.data_test = random_split(full_ds,
                                                                      [0.8,
                                                                       0.1,
                                                                       0.1])


def _process_raw_data(filename: str, dirname: Path):
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall()
    img_dir = dirname / "faces"
    annotation_file = img_dir / "face_landmarks.csv"
    # create a pandas dataframe
    landmarks_df = pd.read_csv(annotation_file)
    imgs = []
    landmarks = []

    for i in range(len(landmarks_df)):
        img = cv2.imread(str(img_dir / landmarks_df.iloc[i, 0]))
        landmark = landmarks_df.iloc[i, 1:]
        landmark_np_arr = np.array([landmark])
        imgs.append(img)
        landmarks.append(landmark_np_arr)
    os.chdir(curdir)
    return imgs, landmarks


if __name__ == "__main__":
    landmark_ds = FaceLandmarksDataset(None)
    landmark_ds.prepare_data()
    landmark_ds.setup()
    print(landmark_ds.data_train[0])
