# %%
import pandas as pd
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import os
import torch

# %% [markdown]
# ### Data Setup

# %%
import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
#print(DATA_PATH)


############################ Data Setup ############################
# %%
# Import Metadata
df = pd.read_json(f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/metadata/dresses_metadata.json").T.reset_index().rename(columns={'index': 'sku'})[['sku', 'garment_type']]
# Create id2label and label2id mappings
id2label = {i:elem for i,elem in enumerate(df.garment_type.value_counts().index)}
label2id = {elem:i for i,elem in enumerate(df.garment_type.value_counts().index)}
# Map labels to ids
df['label'] = df.garment_type.map(label2id)
# Save to disc
pickle.dump(id2label, open(f"garment_type_id2label.pkl", "wb"))

######################## Data Split ############################
# %%
# Train Test Split
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
# Save to disc
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)

########################### Dataset and Data Loader ############################
# %%
class DressCategoriesDataset(Dataset):
    """Dress Categories Dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.dataframe.iloc[idx, 0]+ '.jpg', )
        image = Image.open(img_name).convert('RGB')
        label = self.dataframe.iloc[idx, 2]
        

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}

        return sample


# %%
def setup_data_loaders(train_transform, test_transform, batch_size=32, shuffle=True, num_workers=0):
    train_dataset = DressCategoriesDataset(csv_file='train.csv', root_dir=f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/images/square_images", transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    test_dataset = DressCategoriesDataset(csv_file='test.csv', root_dir=f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/images/square_images", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return train_loader, test_loader


