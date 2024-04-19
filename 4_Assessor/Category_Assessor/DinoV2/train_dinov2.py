# %%
from torchvision import transforms, utils
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification
from glob import glob
from PIL import Image
from tqdm.notebook import tqdm
import torch
import torch.optim as optim
import numpy as np
import os
import logging


from Data_Setup import setup_data_loaders, id2label, label2id

# %%
import platform
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"

# %%
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# %% [markdown]
# ### Initialize Model and Dataloaders

# %%
id2label

# %%
# %%
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id)
model = model.to(device)

mean = processor.image_mean
std = processor.image_std
interpolation = processor.resample

train_transform = Compose([
    #RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=interpolation),
    RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

test_transform = Compose([
    ToTensor(),
    Normalize(mean=mean, std=std),
])

# %% [markdown]
# ### Training Loop

# %%
def train_epoch(model, train_loader, loss_fn, optimizer, report_interval=10):
    model.train()
    running_loss = 0.0
    running_total = 0
    running_correct = 0
    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data['image'], data['label']
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(outputs.logits, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
        running_loss += loss.item()

        if i % report_interval == report_interval - 1:
            print(f'\t[{i + 1}/{len(train_loader)}] loss: {running_loss / report_interval} accuracy: {np.round((running_correct/running_total)*100, 4)}%')
            logging.info(f'\t[{i + 1}/{len(train_loader)}] loss: {running_loss / report_interval} accuracy: {np.round((running_correct/running_total)*100, 4)}%')
            running_loss = 0.0

    return running_loss / len(train_loader)

def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

    import torch
from tqdm import tqdm

def test_model(model, test_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels = data['image'], data['label']
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            
            # Update overall accuracy counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update per class accuracy counts
            for label, prediction in zip(labels, predicted):
                if label.item() not in class_correct:
                    class_correct[label.item()] = 0
                    class_total[label.item()] = 0
                class_correct[label.item()] += (prediction == label).item()
                class_total[label.item()] += 1

    # Calculate overall accuracy
    overall_accuracy = correct / total if total > 0 else 0

    # Calculate per class accuracy
    class_accuracies = {}
    for label in class_correct:
        class_accuracies[label] = (class_correct[label] / class_total[label]
                                   if class_total[label] > 0 else 0)

    return overall_accuracy, class_accuracies


# %%
# Define Hyperparameters
NUM_EPOCHS = 5
LR = 0.001
BATCH_SIZE = 4
NUM_WORKERS = 4

BACKBONE_FROZEN = True

if BACKBONE_FROZEN:
    for param in model.dinov2.parameters():
        param.requires_grad = False

model = model.to(device)

# Define Loss and optimizers
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Model save paths
save_dir = f"{DATA_PATH}/Models/Assessor/DinoV2/batch_size_{BATCH_SIZE}_LR_{LR}_NUM_EPOCHS_{NUM_EPOCHS}/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

train_loader, test_loader = setup_data_loaders(train_transform, test_transform, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# %%
if logging.root.handlers:
    # If there are handlers, clear them
    logging.root.handlers = []
logging.basicConfig(filename=f'{save_dir}log.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')

print('Starting Training...')
# accuracy = test_model(model, test_loader)
# print(f"Initial accuracy: {np.round(accuracy*100, 4)}%")
# logging.warning(f"Initial accuracy:  {np.round(accuracy*100, 4)}%")
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch}:")
    logging.info(f"Epoch {epoch}:")
    train_epoch(model, train_loader, loss_fn, optimizer, report_interval=20)
    # Save the model 
    torch.save(model, f"{save_dir}Epoch_{epoch}")

    # Evaluate the accuracy after each epoch
    accuracy, class_accuracy = test_model(model, test_loader)
    print(f"Validation Accuracy after epoch {epoch}: {np.round(accuracy*100, 2)}%")
    logging.info(f"Validation Accuracy after epoch {epoch}: {np.round(accuracy*100, 2)}%")
    class_accuracies = {id2label[k]:v for k,v in class_accuracy.items()}
    logging.info(dict(sorted(class_accuracies.items(), key=lambda item: item[1])))


