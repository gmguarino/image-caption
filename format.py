from json import load
import fiftyone as fo
import os

from dotenv import load_dotenv
from model import PreprocessorCNN

load_dotenv("PATHS.env")

# Place all dir paths in .env file
# create instance for preprocessor inception model
# create pytorch dataset from fiftyone dataset
# evaluate all images w/ inceptionv3
# Cache images in data folder

def get_preprocessor():
    pass

def get_fiftyone_dataset(name="coco-2017", split="train"):
    dataset = fo.Dataset.from_dir(
        name=name,
        dataset_dir=os.path.join(os.getenv("COCO_PATH"), split),
        # dataset_type=fo.types.COCODetectionDataset
        dataset_type=fo.types.COCODetectionDataset

    )
    return dataset

def preprocess():
    pass

def chache():
    pass