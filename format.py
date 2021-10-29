from json import load
import os

from dotenv import load_dotenv
from torch.utils.data import DataLoader
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
    return loader
    

def preprocess():
    pass

def chache():
    pass


if __name__=="__main__":
    loader = get_dataloader()
    preprocessor = get_preprocessor()
    print(next(loader.__iter__()))