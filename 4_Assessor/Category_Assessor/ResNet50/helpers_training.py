import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet50, ResNet50_Weights
from tqdm.notebook import tqdm
from sklearn.utils import check_random_state

import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"


def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    return device

device = set_device()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    check_random_state(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_data(df, target_feature):
    df = df[['sku', target_feature]].copy()

    # Drop rows with missing values
    df = df[df[target_feature].notna()]

    # Create id2label and label2id mappings
    id2label = {i:elem for i,elem in enumerate(df[target_feature].value_counts().index)}
    label2id = {elem:i for i,elem in enumerate(df[target_feature].value_counts().index)}
    # Map labels to ids
    df['label'] = df[target_feature].map(label2id)
    # Save to disc
    pickle.dump(id2label, open(f"id2label_dicts/{target_feature}_id2label.pkl", "wb"))
    return df, id2label, label2id

class DressCategoriesDataset(Dataset):
    """Dress Categories Dataset"""

    def __init__(self, df, root_dir, transforms=None):
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index_id = self.df.iloc[idx]['index_id']

        sku = self.df.iloc[idx]['sku']
        img = Image.open(f"{self.root_dir}/{sku}.jpg")
        label = self.df[self.df.index_id == index_id]['label'].values[0]

        if self.transforms:
            img = self.transforms(img)
        

        return img, label
    
def get_datasets(df, root_dir, train_transform, test_transform):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
    train = train.reset_index().rename(columns={'index': 'index_id'})
    test = test.reset_index().rename(columns={'index': 'index_id'})

    train_dataset = DressCategoriesDataset(train, root_dir, train_transform)
    test_dataset = DressCategoriesDataset(test, root_dir, test_transform)

    return train_dataset, test_dataset

# Training Loop: 
def train_epoch(model, train_loader, test_loader, criterion, optimizer, epoch_num, report_interval, eval_every):

    # Configure how often to evaluate based on the total images shown
    eval_steps = max(1, eval_every // train_loader.batch_size)  

    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0
    for i, data in enumerate(tqdm(train_loader, leave=False, desc=f"Epoch {epoch_num}")):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        running_corrects += (predicted == labels).sum().item()
        running_total += labels.size(0)

        if i % report_interval == report_interval-1:

            print(f"Batch {i+1}: LOSS within report interval: {running_loss / report_interval} | ACCURACY within report interval: {running_corrects / running_total}")
            running_loss = 0.0

        if i % eval_steps == eval_steps - 1:
            train_acc = evaluate(model, train_loader)
            test_acc = evaluate(model, test_loader)
            print(f"Batch {i+1}: Train Accuracy: {train_acc} | Test Accuracy: {test_acc}") 
        
    
    # Evaluate model
    train_acc = evaluate(model, train_loader)
    test_acc = evaluate(model, test_loader)
    print(f"After Epoch {epoch_num}: Train Accuracy: {train_acc} | Test Accuracy: {test_acc}")


# Evaluation function: 
def evaluate(model, data_loader):
    print("Evaluating...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        #for data in tqdm(data_loader, leave=False, desc="Evaluating"):
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_dataset, test_dataset, criterion, optimizer, epochs, batch_size, report_interval=20, eval_every=1000, save_dir=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        train_epoch(model, train_loader, test_loader, criterion, optimizer, epoch+1, report_interval, eval_every)
        if save_dir:
            torch.save(model, f"{save_dir}model_epoch_{epoch+1}.pt")