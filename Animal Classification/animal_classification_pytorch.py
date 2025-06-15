# animal_classification_pytorch.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# Paths and parameters
DATA_DIR = "/teamspace/studios/this_studio/Projects/Animal Classification/dataset"
BATCH_SIZE = 32
NUM_WORKERS = 4
INPUT_SIZE = 224
NUM_CLASSES = 15
EPOCHS = 10
FINE_TUNE_EPOCHS = 5
SPLIT_RATIO = [0.7, 0.15, 0.15]  # train, val, test

# Data transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load full dataset with no transforms to split
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transforms.ToTensor())

total_size = len(full_dataset)
train_size = int(SPLIT_RATIO[0] * total_size)
val_size = int(SPLIT_RATIO[1] * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Assign transforms to subsets
train_dataset.dataset.transform = train_transforms
val_dataset.dataset.transform = val_test_transforms
test_dataset.dataset.transform = val_test_transforms

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build model using Transfer Learning (ResNet50)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(device)

# Freeze backbone parameters
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# Loss and optimizer (only for classifier head)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Training & validation loops
def train_epoch(loader, model, criterion, optimizer):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

def eval_epoch(loader, model, criterion):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc.item()

# Initial training
best_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_acc = eval_epoch(val_loader, model, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_initial.pth')

# Fine-tuning: unfreeze some layers
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

for epoch in range(EPOCHS, EPOCHS + FINE_TUNE_EPOCHS):
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer)
    val_loss, val_acc = eval_epoch(val_loader, model, criterion)
    print(f"Fine-tune Epoch {epoch+1}/{EPOCHS + FINE_TUNE_EPOCHS} - "
          f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model_finetuned.pth')

# Final evaluation on test set
model.load_state_dict(torch.load('best_model_finetuned.pth'))
test_loss, test_acc = eval_epoch(test_loader, model, criterion)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Save final model
torch.save(model.state_dict(), 'animal_classifier_resnet50.pth')
