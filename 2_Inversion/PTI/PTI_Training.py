#%% 
import os
from PIL import Image

os.chdir('../../PTI/')

#%%
from scripts.run_pti import run_PTI
run_PTI(run_name='whole_sample', use_wandb=False, use_multi_id_training=False)
# %%
