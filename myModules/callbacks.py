###
# file name: callbacks.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

from pytorch_lightning.callbacks import LearningRateMonitor, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def set_callbacks(params):
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    gpu_stats = GPUStatsMonitor(temperature=True)
    checkpoint = ModelCheckpoint(
        monitor='val_acc',
        save_last = True,
        filename='{epoch}_{val_acc:.3f}',
        save_top_k=3,
        mode='max')
    callbacks = [lr_monitor, gpu_stats, checkpoint]
    if params.training.early_stop:
        callbacks.append(EarlyStopping(
            monitor= params.training.early_stop_setting.monitor,
            min_delta= params.training.early_stop_setting.min_delta,
            patience= params.training.early_stop_setting.patience,
            verbose= params.training.early_stop_setting.verbose,
            mode= params.training.early_stop_setting.mode
        ))
    return callbacks