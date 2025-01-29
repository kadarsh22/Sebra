import torch
from collections import OrderedDict
from dataset.bar import BAR
import pickle
import glob
import torchvision
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Define constants
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)
GRID_ROW_COUNT = 10

# Transformation pipeline
normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
rank_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize,
])

# Utility function to visualize high and low rank images for all classes in a single figure
def visualize_all_classes_grid(spuriosity_ranking, dataset, pdf_path, title="All Classes - High and Low Rank Images"):
    """
    Create and save a grid of high and low rank images for all classes from the dataset in a single figure.

    Args:
        spuriosity_ranking (dict): Dictionary with class indices as keys and rankings as values.
        dataset (Dataset): Dataset object to fetch images from.
        pdf_path (str): Path to save the figure as a PDF.
        title (str): Title for the plot.
    """
    all_images = []
    all_annotations = []

    for class_idx, indices in spuriosity_ranking.items():
        indices = np.flip(indices, axis=0)
        high_rank_indices = indices[:10]
        low_rank_indices = indices[-10:]

        high_rank_images = torch.stack([dataset.__getitem__(idx)['image'] for idx in high_rank_indices])
        low_rank_images = torch.stack([dataset.__getitem__(idx)['image'] for idx in low_rank_indices])

        # Append images and annotations
        all_images.extend([high_rank_images, low_rank_images])
        all_annotations.append(f"Class {class_idx}")

    # Plotting
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(len(all_annotations), 1, figsize=(20, len(all_annotations) * 5))
        fig.suptitle(title, fontsize=16)

        for i, class_title in enumerate(all_annotations):
            # Combine high and low rank images vertically
            combined_images = torch.cat((all_images[i * 2], all_images[i * 2 + 1]), dim=0)
            grid = torchvision.utils.make_grid(combined_images, nrow=GRID_ROW_COUNT, normalize=True, padding=2, pad_value=1)
            grid_np = grid.numpy().transpose((1, 2, 0))

            # Display the grid
            axes[i].imshow(grid_np)
            axes[i].axis('off')
            axes[i].set_title(f"{class_title}: High Rank (Top) and Low Rank (Bottom)", fontsize=12)

        # Add white space between classes
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close()

# Load original spuriosity ranking
with open('assets/ranking/bar/neurips23mmoayeri/index.pkl', 'rb') as file:
    spuriosity_ranking_org = pickle.load(file)

spuriosity_ranking = spuriosity_ranking_org['top_idx']['train']
train_no_aug = BAR(split='train', transform=rank_transform, indices=spuriosity_ranking_org['train_idx'])

visualize_all_classes_grid(spuriosity_ranking, train_no_aug, "assets/appendix/bar/spuriosity_rankings.pdf", title="Spuriosity Rankings - All Classes")

# Load and process spuriosity rankings from paths
ranking_paths = glob.glob("exp/ranks_bar/*")[:1]
for path in ranking_paths:
    with open(path, 'rb') as file:
        spuriosity_ranking_ours = pickle.load(file)

    # Compile rankings across classes
    ranking_compiled = {i: [] for i in range(6)}
    for rank_dict in spuriosity_ranking_ours.values():
        for class_idx in range(6):
            ranking_compiled[class_idx].extend(list(OrderedDict.fromkeys(rank_dict[class_idx])))

    # Save high and low rank images for the compiled rankings
    visualize_all_classes_grid(ranking_compiled, train_no_aug, "assets/appendix/bar/sebra_rankings.pdf", title="Sebra Rankings - All Classes")
