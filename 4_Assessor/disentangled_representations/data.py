import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import os
import pickle

import sys
sys.path.append('..')
#from attribute_driven_representations.labels import Labels
from labels import Labels


import platform
import os
if platform.system() == 'Darwin':
    DATA_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Data.nosync"
    ROOT_PATH = "/Users/maltegenschow/Documents/Uni/Thesis/Thesis"
elif platform.system() == 'Linux':
    DATA_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Data.nosync"
    ROOT_PATH = "/pfs/work7/workspace/scratch/tu_zxmav84-thesis/Thesis"

current_wd = os.getcwd()



# TRANSFORMS ------------------------------------------------------------------

### RESNET ###
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((pretrained_size, pretrained_size)),
    transforms.RandomRotation(5),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomCrop(pretrained_size, padding=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

test_valid_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((pretrained_size, pretrained_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

resnet_transforms = {
    'train': train_transforms,
    'test_valid': test_valid_transforms,
}

### DINOv2 ###
transforms_dinov2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(244, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize([0.5], [0.5])
])

dinov2_transforms = {
    'train': transforms_dinov2,
    'test_valid': transforms_dinov2,
}

# Combine in dictionary
transforms = {
    'resnet': resnet_transforms,
    'dinov2': dinov2_transforms
}

# IMAGE UTILS -----------------------------------------------------------------

def make_square(im, resize_size = None):
    """Converts image to square by adding background color."""
    max_size = max(im.size)
    # Infer bckground color from top left pixel
    background_color = im.getpixel((0,0))
    # Create new image with correct background
    new_im = Image.new("RGB", (max_size, max_size), color=background_color)
    # Paste image in center
    new_im.paste(im, box = ((max_size - im.size[0])//2,0))

    # If Resizing so specific size is needed:
    if resize_size:
        new_im = new_im.resize((resize_size, resize_size))

    return new_im


# IMAGE DATA ------------------------------------------------------------------

class Zalando(data.Dataset):
    def __init__(
            self,
            meta_data: pd.DataFrame,
            img_dir: str,
            dataset_name: str,
            transform = None
        ) -> None:
        """Custom class for the Zalando image dataset."""

        self.meta_data = meta_data
        self.img_dir = img_dir
        self.transform = transform
        self.Labels = Labels(dataset_name=dataset_name)
        self.task_label_mapping = self.Labels.task_mapping

        # Store image ids (just to access in get_image_embeddings)
        self.image_ids = self.meta_data['article_id'].tolist()

    def __getitem__(self, index):

        image_metadata = self.meta_data.iloc[index,:]

        # Read image as array
        img_file = f'{image_metadata["article_id"]}.jpg'
        filepath = os.path.join(self.img_dir, img_file)
        image = np.array(Image.open(filepath).convert('RGB'))

        # Transforms
        if self.transform is not None:
            image = self.transform(image)

        # Assemble labels
        labels = [
            self.task_label_mapping[task][image_metadata[task]]
            for task in self.task_label_mapping.keys()
        ]

        return image, labels

    def __len__(self):
        return len(self.meta_data)


class InferenceData(data.Dataset):
    def __init__(
            self,
            img_dir: str,
            square: bool = True,
            embedding_model: str = 'dinov2',
            N: int = None
        ) -> None:
        """Custom class for new image data to make inference on."""

        self.img_dir = img_dir
        self.square = square
        self.transform = transforms[embedding_model]['test_valid']
        self.img_files = os.listdir(img_dir)

        # Subset (for debugging purposes)
        if N is not None:
            self.img_files = self.img_files[:N]

    def __getitem__(self, index):

        # Read image
        img_file = self.img_files[index]
        filepath = os.path.join(self.img_dir, img_file)
        image = Image.open(filepath).convert('RGB')

        # Square
        if self.square:
            image = make_square(image)

        # Transforms
        image = np.array(image)
        if self.transform is not None:
            image = self.transform(image)

        return img_file, image

    def __len__(self):
        return len(self.img_files)


def get_zalando_data(
        dataset_name: str,
        sample: str = 'train',
        img_type: str = 'square',  # Either 'raw' or 'square'
        verbose: bool = True,
        N: int = None,
        model = 'dinov2'  # 'resnet' or 'dinov2' -> needed for transforms
    ):

    # Read metadata from CSV file
    meta_data = pd.read_csv(f'{DATA_PATH}/Models/disentangled_representations/{dataset_name}/labels.csv')
    n_articles = len(meta_data)
    
    # Filter for sample
    meta_data = meta_data.loc[meta_data['sample'] == sample, :]

    # Subset (for debugging purposes)
    if N is not None:
        meta_data = meta_data.iloc[:N, :]
    
    if verbose:
        n_articles_selected = len(meta_data)
        print(
            f'Selected {n_articles_selected} of {n_articles} articles',
            f'({n_articles_selected/n_articles*100:.2f}%)',
            f'for {sample} sample.'
        )

    data = Zalando(
        meta_data=meta_data,
        img_dir=f'{DATA_PATH}/Generated_Images/e4e/00005_snapshot_1200/',
        dataset_name=dataset_name,
        transform=transforms[model][('train' if sample == 'train' else 'test_valid')]
    )

    return data


def get_data(
        dataset_name: str,
        sample: str = 'train',
        verbose: bool = True,
        model: str = 'dinov2',
        **kwargs  # Further arguments for get_data
    ) -> data.Dataset:
    """Get dataset (agnostic of underlying dataset)."""

    kwargs = dict(sample=sample, verbose=verbose, model=model, **kwargs)

    if dataset_name.startswith('zalando'):
        data = get_zalando_data(dataset_name=dataset_name, **kwargs)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return data


# EMBEDDINGS ------------------------------------------------------------------

class Embeddings(data.Dataset):
    def __init__(self, dataset_name: str, model_name: str, sample: str) -> None:
        """Custom class for the embeddings (agnostic of underlying dataset)."""

        # Load data from file
        embeddings_filepath = f'{DATA_PATH}/Models/disentangled_representations/{dataset_name}/embeddings/{model_name}/{sample}.pkl'
        with open(embeddings_filepath, 'rb') as f:
            data = pickle.load(f)

        # Extract embeddings and labels
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.image_ids = data['images_ids']

        # Get mapping class
        self.Labels = Labels(dataset_name=dataset_name)

    def __getitem__(self, index):
        embedding = self.embeddings[index,:]
        labels = [int(task_labels[index]) for task_labels in self.labels.values()]
        return embedding, labels

    def __len__(self):
        return len(self.embeddings)
