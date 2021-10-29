from json import load
import os

from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator

from model import PreprocessorCNN
from dataset import COCOCaptionDataset

load_dotenv("PATHS.env")
COCO_PATH = os.environ.get("COCO_PATH")


# Place all dir paths in .env file
# create instance for preprocessor inception model
# create pytorch dataset from fiftyone dataset
# evaluate all images w/ inceptionv3
# Cache images in data folder


def get_preprocessor():
    
    preprocessor = PreprocessorCNN()
    preprocessor.eval()
    return preprocessor


def get_dataloader():
    dataset = COCOCaptionDataset(COCO_PATH, "train")
    loader = DataLoader(dataset)
    return loader, 
    

def preprocess():
    loader = get_dataloader()
    preprocessor = get_preprocessor()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor.to(device)
    out = next(iter(loader))
    print(len(out),  out )# WHY IS THIS RETURNING A DATALOADER??
    # out_tensor = preprocessor(img_tensor.to(device))
    # print(out_tensor.shape)


def chache():
    pass


if __name__=="__main__":
    preprocess()
