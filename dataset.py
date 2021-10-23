import torch
import torch.nn as nn
from PIL import Image
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
        self.split = "train" if split=="train" else "validation"
        with open(os.path.join(coco_path, "raw", f"captions_{split}2017.json")) as jf:
            self.captions = json.load(jf)["annotations"]
        self.coco_path = coco_path
        self.image_list = os.listdir(os.path.join(COCO_PATH, self.split, "data"))
        # Ǹeeds to be more efficient
        self.captions = [ann for ann in self.captions if f"{ann['image_id']:012}.jpg" in self.image_list]
        super().__init__()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        caption = self.captions[index]["caption"]
        image_id = self.captions[index]["image_id"]
        filepath = os.path.join(self.coco_path, self.split, "data", f"{image_id:012}.jpg")
        print(f"{image_id:012}.jpg" in self.image_list)
        image = Image.open(filepath)
        print(filepath)

        return caption, image


if __name__=="__main__":
    with open(os.path.join(COCO_PATH, "raw", "captions_train2017.json")) as jf:
        captions = json.load(jf)["images"]

    print(captions[1])
    print(len("000000522418"))
    dataset = COCOCaptionDataset(COCO_PATH, "train")
    dataset.__getitem__(0)