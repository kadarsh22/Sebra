import os
import torch
import pickle
import itertools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def create_imagenet_subset(imagenet_dir, subset_indices, subset_dir):
    """
    Create a subset of the ImageNet dataset and save to a new directory.

    Parameters:
    - imagenet_dir: str, path to the full ImageNet dataset
    - subset_indices: list of int, indices of images to include in the subset
    - subset_dir: str, path to save the subset of ImageNet
    """
    # Define transformations (optional, depending on needs)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images if needed
        transforms.ToTensor(),
    ])

    # Load the full ImageNet dataset
    full_imagenet = datasets.ImageNet(root=imagenet_dir, split='train', transform=transform)

    # Create a subset using the provided indices
    imagenet_subset = Subset(full_imagenet, subset_indices)

    # Create DataLoader for the subset for easy processing
    subset_loader = DataLoader(imagenet_subset, batch_size=1, shuffle=False)

    # Create subset directory if it doesn't exist
    os.makedirs(subset_dir, exist_ok=True)

    # Iterate over the subset and save images to the subset directory
    for i, (image, label) in enumerate(subset_loader):
        # Image and label information
        class_name = full_imagenet.wnids[label.item()]
        class_dir = os.path.join(subset_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Save image
        save_path = os.path.join(class_dir, f"image_{subset_indices[i]:05d}.jpg")
        transforms.ToPILImage()(image.squeeze(0)).save(save_path)

    print(f"Subset created and saved at: {subset_dir}")


# Parameters
imagenet_dir = '/vol/research/project_storage/data/imagenet'  # Path to the full ImageNet dataset
with open('assets/ranking/imagenet/spuriosity_ranking/bottom_spurious_imagenet_idx.pkl', 'rb') as file1:
    data_idx = pickle.load(file1)

# with open('assets/ranking/imagenet/spuriosity_ranking/bottom_spurious_imagenet_idx.pkl', 'rb') as file1:
#     data_idx = pickle.load(file1)
subset_idx = list(itertools.chain(*data_idx))

subset_indices = subset_idx  # List of indices to include in the subset
subset_dir = '/vol/research/project_storage/data/imagenet_lowspu'  # Directory to save the subset

# Run function to create subset
create_imagenet_subset(imagenet_dir, subset_indices, subset_dir)
