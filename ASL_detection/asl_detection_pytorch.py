# asl_detection_pytorch.py
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

# Paths and parameters
TRAIN_DIR = os.path.expanduser("/teamspace/studios/this_studio/Projects/ASL_detection/asl_alphabet_train/asl_alphabet_train")
TEST_DIR = os.path.expanduser("/teamspace/studios/this_studio/Projects/ASL_detection/asl_alphabet_test/asl_alphabet_test")
BATCH_SIZE = 32
NUM_WORKERS = 4
INPUT_SIZE = 224
NUM_CLASSES = 29
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Class names (alphabet plus del, nothing, space)
classes = sorted(os.listdir(TRAIN_DIR))  # should list A-Z, del, nothing, space

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = val_transforms

# Datasets
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
# We'll use a portion of train as validation
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_set, val_set = torch.utils.data.random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
val_set.dataset.transform = val_transforms

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_dir, classes, transform=None):
        self.transform = transform
        self.samples = []
        for fname in glob.glob(os.path.join(test_dir, '*_test.jpg')):
            # label is prefix before '_test.jpg'
            label_name = os.path.basename(fname).split('_test.jpg')[0]
            if label_name == 'del':
                label_name = 'del'  # matches folder name
            self.samples.append((fname, classes.index(label_name)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_dataset = TestDataset(TEST_DIR, classes, transform=test_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# Model: Transfer Learning with ResNet18
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# Freeze base layers
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Training and evaluation functions
def train_epoch(loader):
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels)
    return running_loss / len(loader.dataset), correct.double() / len(loader.dataset)

def eval_epoch(loader):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)
    return running_loss / len(loader.dataset), correct.double() / len(loader.dataset)

# Training loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(train_loader)
    val_loss, val_acc = eval_epoch(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
          f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_asl_initial.pth')

# Fine-tuning
for name, param in model.named_parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)
for epoch in range(EPOCHS, EPOCHS + 5):
    train_loss, train_acc = train_epoch(train_loader)
    val_loss, val_acc = eval_epoch(val_loader)
    print(f"Fine-tune Epoch {epoch+1}/{EPOCHS+5} - "
          f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
          f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_asl_finetuned.pth')

# Testing
model.load_state_dict(torch.load('best_asl_finetuned.pth'))

test_loss, test_acc = eval_epoch(test_loader)
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

# Save final model
torch.save(model.state_dict(), 'asl_classifier_resnet18.pth')

