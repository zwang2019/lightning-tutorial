# Lightning has huge benefits for multi-GPU training, and makes code less boilerplate.
from typing import Any

import lightning as L
import torch

from src.config import CONFIG
from src.dataset import MnistDataModule
from src.model import NN
from src.callbacks import MyPrintingCallback, EarlyStopping, Timer, ModelCheckpoint

from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':

    my_dm = MnistDataModule(data_dir=CONFIG.data_dir, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers)
    model = NN(input_size=CONFIG.input_size, num_classes=CONFIG.num_classes, learning_rate=CONFIG.learning_rate)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min')
    timer = Timer()
    checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, verbose=True, dirpath='./models', filename='{epoch:02d}-{val_accuracy:.6f}')

    # Logger
    logger = TensorBoardLogger(save_dir='./tensorboardlogger', name='tutorial_model_v0', version=1)

    trainer = L.Trainer(logger=logger, accelerator='gpu', devices=1, strategy='auto', max_epochs=CONFIG.num_epochs, callbacks=[checkpoint_callback, early_stopping, timer])
    trainer.fit(model, my_dm)
    trainer.test(model, my_dm)

    # Print the time taken for each stage
    print('time taken for training: %.2f s' % timer.time_elapsed("train"))
    print('time taken for validation: %.2f s' % timer.time_elapsed("validate"))
    print('time taken for testing: %.2f s' % timer.time_elapsed("test"))

    # Print the best model path
    print('best model path: %s' % checkpoint_callback.best_model_path)
    # Print the best model score
    print('best model score: %.6f' % checkpoint_callback.best_model_score)

