import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class GrayscaleColorDataset(Dataset):
    def __init__(self, train=True):
        # Load CIFAR-10 dataset and apply resizing and normalization
        self.data = datasets.CIFAR10(
            root='./data',
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        color_img, _ = self.data[idx]  # RGB image, shape: [3, 64, 64]

        # Convert to grayscale using luminance formula
        r, g, b = color_img[0], color_img[1], color_img[2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b  # shape: [64, 64]
        gray = gray.unsqueeze(0)  # shape: [1, 64, 64]

        return gray, color_img  # (input, target)
