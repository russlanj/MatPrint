import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image
import os

# Custom dataset
class ImageRegressionDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image (convert to grayscale)
        image = Image.open(img_path).convert("L")  # "L" mode is for grayscale
        
        # Extract label from the filename (assuming the filename is the target value)
        label = float(os.path.splitext(img_name)[0])
        
        if self.transform:
            image = self.transform(image)
        
        # Ensure label is float32
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label

# Transformations for grayscale images
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),  # Adjust mean and std for single-channel images
])

# Load dataset
img_dir = r'./form_en_values' #Please read the README file to correctly use the dataset
dataset = ImageRegressionDataset(img_dir, transform=transform)

# Split dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept single-channel (grayscale) images
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify the final layer to output a single value
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

for param in model.parameters():
    param.requires_grad = True

# Move model to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print the name of the device being used
print(f"Using device: {device}")

# Define loss function and optimizer
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 220

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)  # Ensure labels are the right shape

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print("Finished Training")

# Evaluate on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)

# Calculate the average test loss
test_loss /= len(test_dataset)
print(f'Test Loss: {test_loss:.4f}')
