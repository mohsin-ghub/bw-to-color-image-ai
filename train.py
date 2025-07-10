import torch
from torch.utils.data import DataLoader
from custom_dataset import GrayscaleColorDataset
from colorization_model import ColorizationNet
import os

# ==== Step 1: Setup device ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==== Step 2: Load Dataset ====
train_dataset = GrayscaleColorDataset(train=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==== Step 3: Load Model ====
model = ColorizationNet().to(device)

# ==== Step 4: Define Loss & Optimizer ====
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ==== Step 5: Training Loop ====
num_epochs = 10  # You can increase this later

for epoch in range(num_epochs):
    total_loss = 0.0
    model.train()

    for batch_idx, (gray, color) in enumerate(train_loader):
        gray = gray.to(device)
        color = color.to(device)

        # Forward pass
        outputs = model(gray)
        loss = criterion(outputs, color)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # ==== Step 6: Save checkpoint ====
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/colorization_epoch_{epoch+1}.pth")

print("✅ Training complete!")
