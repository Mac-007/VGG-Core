'''
with Early Stopping

@ Dr. Amit Chougule - 25/07/2025
'''

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

# Set paths
data_dir = "<Data_Dir>"
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transformations
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform['train'])
print(train_dataset.class_to_idx)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Load VGG16 pretrained model
model = models.vgg16(pretrained=True)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Modify the classifier for binary classification
model.classifier[6] = nn.Linear(4096, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

# Early stopping parameters
patience = 5
early_stopping_counter = 0
best_val_acc = 0.0

# Training loop
num_epochs = 100
train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_acc = 100 * correct / total
    train_acc_list.append(train_acc)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}% "
          f"Loss: {running_loss/len(train_loader):.4f}")

    # Early Stopping Check
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        early_stopping_counter = 0
        # Save the best model
        torch.save(model.state_dict(), "vgg16_best_model.pth")
        print(" Validation accuracy improved. Model saved.")
    else:
        early_stopping_counter += 1
        print(f" Validation accuracy did not improve. Early stopping counter: {early_stopping_counter}/{patience}")
        if early_stopping_counter >= patience:
            print(" Early stopping triggered.")
            break

# Save final model (last epoch, not necessarily the best)
torch.save(model.state_dict(), "vgg16_image_classifier.pth")
print(" Final model saved to vgg16_image_classifier.pth")

# Plot accuracy
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Validation Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig("accuracy_plot.png")
print(" Accuracy plot saved to accuracy_plot.png")

