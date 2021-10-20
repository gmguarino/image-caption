import fiftyone as fo

# Print your current config
print(fo.config)

# Print a specific config field
print(fo.config.default_ml_backend)


dataset = fo.zoo.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections", "segmentations"],
    max_samples=0000
)
dataset.persistent = True

session = fo.launch_app(dataset)