# %%
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt

# %%
import pickle
import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()

torch.manual_seed(42)

# %%
def set_device():
    try:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    except:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using {device} as device")
    return device

# %%
def get_datasets(target):
    df = pd.read_json(f"{DATA_PATH}/Zalando_Germany_Dataset/dresses/metadata/dresses_metadata.json").T.reset_index().rename(columns={'index': 'sku'})
    df = df[['sku', target]].copy()

    # Drop rows with missing values
    df = df[df[target].notna()].reset_index(drop=True)

    # Load or create id2label and label2id mappings
    if not os.path.exists(f'id2label_dicts/{target}_id2label.pkl'):
        id2label = {i:elem for i,elem in enumerate(df[target].value_counts().index)}
        label2id = {elem:i for i,elem in enumerate(df[target].value_counts().index)}
        pickle.dump(id2label, open(f"id2label_dicts/{target}_id2label.pkl", "wb"))
        pickle.dump(label2id, open(f"id2label_dicts/{target}_label2id.pkl", "wb"))
    else:
        id2label = pickle.load(open(f"id2label_dicts/{target}_id2label.pkl", "rb"))
        label2id = pickle.load(open(f"id2label_dicts/{target}_label2id.pkl", "rb"))

    # Map labels to ids
    df['label'] = df[target].map(label2id)
    
    # Split data into train and test
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)

    return train, test, id2label, label2id

# %%
class Categories_Dataset(Dataset):
    
    def __init__(self, df, target, embeddings_path, id2label, label2id):
        self.df = df
        self.target = target
        self.embedding_path = embeddings_path
        self.id2label = id2label
        self.label2id = label2id
        # Load embeddings
        self.embeddings = torch.load(self.embedding_path, map_location='cpu')

        # Subset df to available embeddings (only relevant for PTI)
        self.df = self.df[self.df['sku'].isin(list(self.embeddings.keys()))].reset_index(drop=True)
        # Subet embeddings to skus in df
        self.embeddings = {sku:embedding for sku, embedding in self.embeddings.items() if sku in list(self.df['sku'])}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        sku = self.df.iloc[idx]['sku']
        label = self.df.iloc[idx]['label']
        embedding = self.embeddings[sku]
        return embedding.squeeze(0), label, sku

# %%
def prepare_data(target, embeddings_name):
    # Load and split data
    train_metadata, test_metadata, id2label, label2id = get_datasets(target)

    # Create torch Datasets
    embeddings_base_path = f"{DATA_PATH}/Models/Assessor/DinoV2/Embeddings"
    train = Categories_Dataset(train_metadata, target, f"{embeddings_base_path}/{embeddings_name}.pt", id2label, label2id)
    test = Categories_Dataset(test_metadata, target, f"{embeddings_base_path}/{embeddings_name}.pt", id2label, label2id)

    return train, test


class ClassifierModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.layer_stack(x)
        return x

class ClassifierModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassifierModel, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.layer_stack(x)
        return x


def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            embeddings, labels, sku = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, NUM_EPOCHS, optimizer, loss_fn, train_loader, test_loader, save_path, device, log_every=1):

    initial_test_acc = evaluate_model(model, test_loader, device)
    initial_train_acc = evaluate_model(model, train_loader, device)
    print(f"Initial Train Accuracy: {initial_train_acc}, Initial Test Accuracy: {initial_test_acc}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        i = 0
        for embeddings, labels, sku in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            i += 1
            if i % log_every == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i}/{len(train_loader)}, Loss: {loss.item()}")

        # Training Set accuracy
        train_acc = evaluate_model(model, train_loader, device)

        # Test Set accuracy
        test_acc = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

    #torch.save(model, f"{save_path}")



def train_model(model, NUM_EPOCHS, optimizer, loss_fn, train_loader, test_loader, save_path, device, log_every=1):
    loss_log = []
    train_acc_log = []
    test_acc_log = []

    initial_test_acc = evaluate_model(model, test_loader, device)
    initial_train_acc = evaluate_model(model, train_loader, device)
    print(f"Initial Train Accuracy: {initial_train_acc}, Initial Test Accuracy: {initial_test_acc}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        i = 0
        for embeddings, labels, sku in train_loader:
            optimizer.zero_grad()
            outputs = model(embeddings.to(device))
            loss = loss_fn(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            i += 1
            if i % log_every == 0:
                loss_log.append(loss.item())  # Log the loss

        # Evaluate accuracies after each epoch
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

    # Save the model
    if save_path:
        torch.save(model, f"{save_path}")

    # Plotting the loss curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_log, label='Loss')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Loss Curve over Training')
    plt.legend()

    # Plotting the accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_log, label='Training Accuracy', marker='o')
    plt.plot(test_acc_log, label='Validation Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve over Training')
    plt.legend()

    plt.tight_layout()
    plt.show()


