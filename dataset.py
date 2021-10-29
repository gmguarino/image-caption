import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

from dotenv import load_dotenv
from time import time
import json
import os

from torchvision import transforms
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

        # setup pytorch tokenizer and a TDIDF vectorizer using maximum 5000 features
        self.tokenizer = get_tokenizer("basic_english")
        self.tdif = TfidfVectorizer(tokenizer=self.tokenizer, max_features=5000, 
            stop_words=list('!"#$%&()*+.,-/:;=?@[\]^_`{|}~'))

        # Processing of caption data
        # Stripping unnecessary characters 
        # self.annotations["vector_caption"] = self.annotations["caption"].map(
        #     lambda x: x.strip('!"#$%&()*+.,-/:;=?@[\]^_`{|}~')
        # )
        # Vectorize
        self.annotations["vector_caption"] = self.tdif.fit_transform(self.annotations["caption"])
        self.caption_tranforms = transforms.ToTensor()
        self.image_transforms = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        super().__init__()

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def spy_sparse2torch_sparse(data):
        """

        :param data: a scipy sparse csr matrix
        :return: a sparse torch tensor
        """
        samples=data.shape[0]
        features=data.shape[1]
        values=data.data
        coo_data=data.tocoo()
        indices=torch.LongTensor([coo_data.row,coo_data.col])
        t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
        return t

    def __getitem__(self, index):
        caption = self.annotations.loc[index, "vector_caption"]
        image_file = self.annotations.loc[index, "image_file"]
        filepath = os.path.join(self.coco_path, self.split, "data", image_file)
        # return caption and PIL image
        image = Image.open(filepath)
        return caption, self.image_transforms(image)


if __name__=="__main__":
    
    dataset = COCOCaptionDataset(COCO_PATH, "train")
    print(dataset.__getitem__(0))
    