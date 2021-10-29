import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

from dotenv import load_dotenv
from time import time
import json
import os
load_dotenv("PATHS.env")

COCO_PATH = os.environ.get("COCO_PATH")


class COCOCaptionDataset(Dataset):

    def __init__(self, coco_path:str, split:str) -> None:
        # Choose if data split
        if split not in ["train", "val"]:
            raise ValueError("Split must be 'train' | 'val'.")
        self.split = "train" if split=="train" else "validation"

        # Annotations saved as list of dictionaries
        with open(os.path.join(coco_path, "raw", f"captions_{split}2017.json")) as jf:
            annotations = json.load(jf)["annotations"]
            # Placing annotations in dataframe
            self.annotations = pd.DataFrame.from_records(annotations)

        self.coco_path = coco_path
        self.image_list = os.listdir(os.path.join(COCO_PATH, self.split, "data"))
        self.annotations["image_file"] = self.annotations["image_id"].map(lambda x:f"{x:012}.jpg")
        # Selection of the annotations for images present in the dataset
        self.annotations = self.annotations.loc[self.annotations.image_file.isin(self.image_list)].reset_index()

        super().__init__()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        caption = self.annotations.loc[index, "caption"]
        image_file = self.annotations.loc[index, "image_file"]
        filepath = os.path.join(self.coco_path, self.split, "data", image_file)
        # return caption and PIL image
        image = Image.open(filepath)
        return caption, image


if __name__=="__main__":
    
    dataset = COCOCaptionDataset(COCO_PATH, "train")
    print(dataset.__getitem__(0))
    