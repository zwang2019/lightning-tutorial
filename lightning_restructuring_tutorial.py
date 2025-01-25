# Lightning has huge benefits for multi-GPU training, and makes code less boilerplate.
from typing import Any

import lightning as L
import torch

from src.config import CONFIG
from src.dataset import MnistDataModule
from src.model import NN

if __name__ == '__main__':

    my_dm = MnistDataModule(data_dir=CONFIG.data_dir, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers)
    model = NN(input_size=CONFIG.input_size, num_classes=CONFIG.num_classes, learning_rate=CONFIG.learning_rate)

    trainer = L.Trainer(accelerator='gpu', devices=1, strategy='auto', max_epochs=CONFIG.num_epochs)
    trainer.fit(model, my_dm)
    trainer.test(model, my_dm)

