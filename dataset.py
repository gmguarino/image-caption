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
        if split not in ["train", "val"]:
            raise ValueError("Split must be 'train' | 'val'.")
        self.split = "train" if split=="train" else "validation"
        with open(os.path.join(coco_path, "raw", f"captions_{split}2017.json")) as jf:
            captions = json.load(jf)["annotations"]
            self.captions = pd.DataFrame.from_records(captions)
        self.coco_path = coco_path
        self.image_list = os.listdir(os.path.join(COCO_PATH, self.split, "data"))
        # Ǹeeds to be more efficient
        self.captions["image_file"] = self.captions["image_id"].map(lambda x:f"{x:012}.jpg")
        self.captions = self.captions.loc[self.captions.image_file.isin(self.image_list)].reset_index()
        # self.captions = [ann for ann in self.captions if f"{ann['image_id']:012}.jpg" in self.image_list]
        
        super().__init__()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        caption = self.captions.loc[index, "caption"]
        image_file = self.captions.loc[index, "image_file"]
        filepath = os.path.join(self.coco_path, self.split, "data", image_file)
        image = Image.open(filepath)
        print(filepath)

        return caption, image


if __name__=="__main__":
    with open(os.path.join(COCO_PATH, "raw", "captions_train2017.json")) as jf:
        captions = json.load(jf)["images"]

    print(captions[1])
    print(len("000000522418"))
    t=time()
    dataset = COCOCaptionDataset(COCO_PATH, "train")
    print("Time", time()-t)
    print(dataset.__getitem__(0))
    # print(dataset.captions.index)