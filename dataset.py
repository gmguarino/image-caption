import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from dotenv import load_dotenv
import json
import os
load_dotenv("PATHS.env")

COCO_PATH = os.environ.get("COCO_PATH")

class COCOCaptionDataset(Dataset):

    def __init__(self, coco_path:str, split:str) -> None:
        if split not in ["train", "val"]:
            raise ValueError("Split must be 'train' | 'val'.")
        self.split = split
        with open(os.path.join(coco_path, "raw", f"captions_{split}2017.json")) as jf:
            self.captions = json.load(jf)["annotations"]
        super().__init__()

    def __len__(self):
        key = "train" if self.split=="train" else "validation"
        return len(os.listdir(os.path.join(COCO_PATH, key, "data")))

    def __getitem__(self, index):
        return super().__getitem__(index)



with open(os.path.join(COCO_PATH, "raw", "captions_train2017.json")) as jf:
    captions = json.load(jf)["annotations"]

print(captions.__len__())