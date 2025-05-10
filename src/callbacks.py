import lightning as L
from jupyter_server.auth import passwd
from lightning.pytorch.callbacks import EarlyStopping, Callback, ModelCheckpoint, Timer

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module) -> None:
        print("Training is starting!")

    def on_train_end(self, trainer, pl_module) -> None:
        print("Training is done!")


