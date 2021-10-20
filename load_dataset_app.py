import fiftyone as fo

# Print your current config
print(fo.config)

# Print a specific config field
print(fo.config.default_ml_backend)
dataset_dir = "/media/giuseppe/Volume/fiftyone/coco-2017/train"

dataset = fo.Dataset.from_dir(
    name="coco-2017",
    dataset_dir=dataset_dir,
    dataset_type=fo.types.COCODetectionDataset
)
session = fo.launch_app(dataset)
session.wait()