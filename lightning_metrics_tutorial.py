# Lightning has huge benefits for multi-GPU training, and makes code less boilerplate.
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
import torchmetrics
from torchmetrics import Metric, Accuracy, F1Score

class NN(L.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1) # flatten
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True,  sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds


    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)





if __name__ == '__main__':

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 784
    num_classes = 10
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 3

    # Load Data
    entire_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
    train_ds, val_ds = random_split(entire_dataset, [50000, 10000])
    test_ds = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)

    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NN(input_size=input_size, num_classes=num_classes)
    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=num_epochs)
    trainer.fit(model, train_loader, val_loader)







