###
# file name: training.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

from myModules.runcontrol import LitModel
from myModules.datawork import wavecaseDataModule
from myModules.callbacks import set_callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin

def _shared_trainer_step(params):
    callbacks = set_callbacks(params)
    if params.training.gpus == 1 or params.training.gpus == 0:
        trainer = Trainer(gpus=params.training.gpus, max_epochs=params.training.max_epoch, precision=params.training.precision,callbacks=callbacks)
    else:
        trainer = Trainer(gpus=params.training.gpus, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False), max_epochs=params.training.max_epoch, precision=params.training.precision,callbacks=callbacks)
    return trainer


def training_model(params):
    wavecase_dm = wavecaseDataModule(params)
    wavecase_dm.prepare_data()
    wavecase_dm.setup()  
    model = LitModel(params)
    trainer = _shared_trainer_step(params)
    trainer.fit(model, wavecase_dm)
    return wavecase_dm, model