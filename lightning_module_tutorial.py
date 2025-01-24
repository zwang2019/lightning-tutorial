# Lightning has huge benefits for multi-GPU training, and makes code less boilerplate.
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim



class NN(L.LightningModule):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def on_train_epoch_end(self) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)









