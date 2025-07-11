import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from custom_dataset import GrayscaleColorDataset
from colorization_model import ColorizationNet
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


test_dataset = GrayscaleColorDataset(train=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

model = ColorizationNet().to(device)
model.load_state_dict(torch.load("models/colorization_epoch_10.pth", map_location=device))
model.eval()

with torch.no_grad():
    for gray, real_color in test_loader:
        gray = gray.to(device)
        real_color = real_color.to(device)

        predicted_color = model(gray)
        break  

def show_images(grays, predicted_colors, real_colors):
    num_images = grays.shape[0]
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        
        plt.subplot(3, num_images, i + 1)
        plt.imshow(grays[i][0].cpu(), cmap='gray')
        plt.title("Input (Gray)")
        plt.axis("off")

        
        plt.subplot(3, num_images, num_images + i + 1)
        plt.imshow(predicted_colors[i].permute(1, 2, 0).cpu())
        plt.title("Predicted (Color)")
        plt.axis("off")

        
        plt.subplot(3, num_images, 2 * num_images + i + 1)
        plt.imshow(real_colors[i].permute(1, 2, 0).cpu())
        plt.title("Ground Truth")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


show_images(gray, predicted_color, real_color)