import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from model import LitAutoEncoder

import pytorch_lightning as pl


if __name__=="__main__":
    # Load dataset with cached data
    dataset = None

    #Randomly split between train and val
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # Create data loaders
    train_loader = DataLoader(mnist_train, batch_size=32)
    val_loader = DataLoader(mnist_val, batch_size=32)

    # model
    model = LitAutoEncoder()

    # training
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, val_loader)
        
