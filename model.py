import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.linear import Linear
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


# Bisogna preparare il modello
# Preparare i gli output
# generare la funzione di caching
# mettere in cache le previsioni
# Training


class PreprocessorCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.preprocess = transforms.Compose([
        #     transforms.Resize(299),
        #     transforms.CenterCrop(299),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        # Overriding fc layer with Identity layer
        self.backbone.fc = Identity()

    def forward(self, x):
        # return self.backbone(self.preprocess(x))
        return self.backbone(x)



class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 3))
        self.decoder = nn.Sequential(
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 28 * 28))

    def forward(self, x):
        embedding = self.encoder(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)    
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)


if __name__=="__main__":
    # data
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=32)
    val_loader = DataLoader(mnist_val, batch_size=32)

    # model
    model = LitAutoEncoder()

    # training
    trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
    trainer.fit(model, train_loader, val_loader)
        
