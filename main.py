###
# file name: main.py (Wavemotion-lightning)
# Author: Dr. Xiaoxian Guo @ Shanghai Jiao Tong University, China
# Email: xiaoxguo@sjtu.edu.cn
# Website: https://naoce.sjtu.edu.cn/teachers/9004.html
# Github: https://github.com/XiaoxG/waveMotion-lightning/
# Create date: 2021/10/08
# Wavemotion-lightning-- a demonstration code for wave motion prediction of an offshore platform. 
# Refers to preprint Arvix: "Probabilistic prediction of the heave motions of a semi-submersible by a deep learning model"
###

# %% 
# Training the model
# Only few data was provided for code demonstration only.

from myModules.training import training_model
from omegaconf import OmegaConf

PARAMS = OmegaConf.load('settings.yaml')

wavecase_dm, model = training_model(PARAMS)

test_loader = wavecase_dm.test_dataloader(noise_level=0.2)
sd_surge, sd_wave, mean_surge, mean_wave = wavecase_dm.nor_para

# %%
import numpy as np
import torch, random
import matplotlib.ticker as plticker
from matplotlib import pyplot as plt

forward_step = model.hparams.data.forward_step
x1 = (np.arange(0,forward_step*3)-forward_step*3)*0.774
x2 = (np.arange(forward_step*3,forward_step*4)-forward_step*3)*0.774

batch = next(iter(test_loader))
model.eval()
with torch.no_grad():
    x_val, y_val = batch
    for i in range(2048):
            n = random.randint(0, 119)
            x_val[i, n, 0] *= 50   # data was in model scale (1:50)
    preds = model(x_val)
# %%
xloc = plticker.MaxNLocator(5)
yloc = plticker.LinearLocator(5)
fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(8, 3.8))
for j in range(2):
    for i in range(4):
        n = random.randint(0, 511)
        axes[j][i].plot(x1, x_val.cpu().numpy()[n,:,0]*sd_surge+mean_surge,color="C7",label='Input')
        axes[j][i].plot(x2, y_val.cpu().numpy()[n,:]*sd_surge+mean_surge,color="C0",label='Truth')
        axes[j][i].plot(x2, preds.cpu().detach().numpy()[n,:]*sd_surge+mean_surge, '.C3', markersize=3, markevery=1, label='Prediction')
        axes[j][i].set_ylim((-6,6))
        axes[j][i].xaxis.set_major_locator(xloc)
        axes[j][i].yaxis.set_major_locator(yloc)
for j in range(2):
    axes[j][0].set_ylabel('Heave (m)')
for i in range(4):
    axes[1][i].set_xlabel('Time (s)')
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, columnspacing=1.5, handletextpad=0.4,bbox_to_anchor=(0.5, 1.05),fancybox=False,edgecolor='k')
fig.tight_layout()
fig.show()