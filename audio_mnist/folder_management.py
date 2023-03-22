#%%
import sys,os
import shutil
import pandas as pd
from tqdm.notebook import tqdm
#%%
annotation_file = open('annotation.csv','w')
annotation_file.write('"label","path"\n')
#%%
data_root = r'C:\Users\Administrator\Desktop\Dataset\audio_mnist'
dataset_root = data_root+r'\dataset'
folder_list = os.listdir(data_root)

# %%
for i in range(10):
    path = os.path.join(dataset_root,str(i))
    try:
        os.makedirs(path)
    except:
        pass
# %%
for folder in tqdm(folder_list):
    if folder[0] not in ['0','1','2','3','4','5','6']:
        continue
    root = os.path.join(data_root,folder)
    audio_list = os.listdir(root)
    
    for file in audio_list:
        file_path = os.path.join(root,file)
        target_path = os.path.join(dataset_root,file[0])
        shutil.copy(file_path,target_path)
        annotation_file.write(f'{file[0]},{os.path.join(target_path,file)}\n')
# %%
annotation_file.close()
# %%
tmp =  pd.read_csv('annotation.csv')
# %%
