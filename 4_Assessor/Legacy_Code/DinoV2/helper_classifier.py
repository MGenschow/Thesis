# %%
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import wandb
import numpy as np
import random
from sklearn.utils import check_random_state


from helper_DinoV2_Embeddings import extract_embeddings, set_device

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
    check_random_state(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# %%
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


# %%
class DressCategoriesDataset(Dataset):
    """Dress Categories Dataset"""

    def __init__(self, df, embeddings):
        self.df = df
        self.embeddings = embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        index_id = self.df.iloc[idx]['index_id']

        label = self.df[self.df.index_id == index_id]['label'].values[0]
        embedding = self.embeddings[index_id,:]

        return embedding, label


# %%
def get_datasets(df, embeddings):
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.label)
    train = train.reset_index().rename(columns={'index': 'index_id'})
    test = test.reset_index().rename(columns={'index': 'index_id'})

    train_dataset = DressCategoriesDataset(train, embeddings)
    test_dataset = DressCategoriesDataset(test, embeddings)

    return train_dataset, test_dataset

# %% [markdown]
# 

# %%
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
    
def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            embeddings, labels = data
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# %%
def train_model(model, NUM_EPOCHS, BATCH_SIZE, optimizer, loss_fn, train_dataset, test_dataset, save_path, device):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Training Set accuracy
        train_acc = evaluate_model(model, train_loader, device)

        # Test Set accuracy
        test_acc = evaluate_model(model, test_loader, device)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss.item()}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")

    torch.save(model, f"{save_path}")

# %%


#################### Weights and Biases Tuning ####################
def train_wb(model_input_dim, model_output_dim, train_dataset, test_dataset, device):
    set_seed(42)
    wandb.init()

    config = wandb.config
    model = ClassifierModel(model_input_dim, model_output_dim)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate and log metrics with W&B
        train_acc = evaluate_model(model, train_loader, device)
        test_acc = evaluate_model(model, test_loader, device)
        wandb.log({'epoch': epoch, 'average_loss': total_loss / len(train_loader), 'train_accuracy': train_acc, 'test_accuracy': test_acc})

    wandb.finish()

def get_optimizer(model, config):
    if config.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)


    



