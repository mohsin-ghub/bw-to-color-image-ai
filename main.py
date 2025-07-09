from custom_dataset import GrayscaleColorDataset
from colorization_model import ColorizationNet
import torch
from torch.utils.data import DataLoader

# Load dataset
train_dataset = GrayscaleColorDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load model
model = ColorizationNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print model and one sample shape
print(model)

# Check shape of a sample batch
gray, color = next(iter(train_loader))
print("Gray shape: ", gray.shape)
print("Color shape:", color.shape)
