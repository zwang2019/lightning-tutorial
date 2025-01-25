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
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()
        self.lr = learning_rate
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = F1Score(task='multiclass', num_classes=num_classes)
        self.training_step_y_hat = []
        self.training_step_y = []
        self.validation_step_y_hat = []
        self.validation_step_y = []

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
        self.training_step_y_hat.append(y_hat)
        self.training_step_y.append(y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.validation_step_y_hat.append(y_hat)
        self.validation_step_y.append(y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True,  sync_dist=True)
        self.validation_step_y_hat.append(y_hat)
        self.validation_step_y.append(y)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size(0), -1)
        y_hat = self(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds

    def on_train_epoch_end(self) -> None:
        y_hat = torch.cat(self.training_step_y_hat)
        y = torch.cat(self.training_step_y)
        # do something with all training_step outputs
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        self.log_dict({'train_loss': loss, 'train_accuracy': acc, 'train_f1_score': f1}, on_epoch=True, prog_bar=True, sync_dist=True)
        # free up the memory
        self.training_step_y_hat.clear()
        self.training_step_y.clear()
        del y_hat, y
        return None

    def on_validation_epoch_end(self) -> None:
        y_hat = torch.cat(self.validation_step_y_hat)
        y = torch.cat(self.validation_step_y)
        # do something with all validation_step outputs
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        self.log_dict({'val_loss': loss, 'val_accuracy': acc, 'val_f1_score': f1}, on_epoch=True, prog_bar=True, sync_dist=True)
        # free up the memory
        self.validation_step_y_hat.clear()
        self.validation_step_y.clear()
        del y_hat, y
        return None

    def on_test_epoch_end(self) -> None:
        y_hat = torch.cat(self.validation_step_y_hat)
        y = torch.cat(self.validation_step_y)
        # do something with all validation_step outputs
        loss = F.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1_score(y_hat, y)
        self.log_dict({'test_loss': loss, 'test_accuracy': acc, 'test_f1_score': f1}, on_epoch=True, prog_bar=True, sync_dist=True)
        # free up the memory
        self.validation_step_y_hat.clear()
        self.validation_step_y.clear()
        del y_hat, y
        return None

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)