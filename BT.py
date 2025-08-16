
"""
BT.py is the python code implemented by Er.Panchanan Nath

Full end-to-end PyTorch training script for brain tumor classification.
Implements 5 custom models (no pretrained): SimpleCNN, VGGLike, ResNetLike, SimpleDenseNetLike, MobileNetLite-like.
Includes:
 - Imports
 - Data loading from Training/ and Testing/ folders using ImageFolder
 - Data visualization (sample images)
 - Training + Validation loops with GPU support
 - Testing and evaluation (accuracy, per-class precision/recall/F1, confusion matrix)
 - Plots (loss/accuracy curves, confusion matrix, sample predictions)
 - Save models (.pt) and histories (.csv)
 - Save plot images

Usage:
 - Set DATA_DIR variable to root folder containing Training/ and Testing/
 - Run on GPU-enabled machine / Colab
"""

import os
import random
import time
import copy
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils

from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns

# ---------------------------
# Config / Hyperparameters
# ---------------------------
DATA_DIR = "./"   # root containing Training/ and Testing/
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR  = os.path.join(DATA_DIR, "Testing")





OUTPUT_DIR = "./results_brain_tumor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 4  # pituitary, notumor, meningioma, glioma
BATCH_SIZE = 32
NUM_EPOCHS = 500
IMAGE_SIZE = 224
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
STEP_LR_STEP = 7
STEP_LR_GAMMA = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Reproducibility
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------------------------
# Data transforms & loaders
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset  = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# Create a small validation split from the training dataset
val_ratio = 0.1
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(val_ratio * num_train))
random.shuffle(indices)
train_idx, val_idx = indices[split:], indices[:split]

from torch.utils.data.sampler import SubsetRandomSampler
train_sampler = SubsetRandomSampler(train_idx)
val_sampler   = SubsetRandomSampler(val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_dataset.classes
print("Class names:", class_names)
print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_dataset)}")

# ---------------------------
# Utility functions for visualization, metrics, saving
# ---------------------------
def imshow_tensor(img_tensor, title=None, mean=None, std=None):
    # img_tensor: CxHxW
    if mean is None: mean = [0.485, 0.456, 0.406]
    if std is None: std = [0.229, 0.224, 0.225]
    img = img_tensor.numpy().transpose((1,2,0))
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title: plt.title(title)
    plt.axis('off')

def show_batch(loader, classes, nrow=6, title="Sample batch"):
    inputs, labels = next(iter(loader))
    grid = utils.make_grid(inputs[:nrow*nrow], nrow=nrow)
    plt.figure(figsize=(12,12))
    imshow_tensor(grid, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    plt.title(title)
    plt.show()

def save_history_csv(history, filepath):
    # history: dict with lists
    keys = list(history.keys())
    rows = list(zip(*[history[k] for k in keys]))
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)

def plot_history(history, model_name, outdir):
    epochs = np.arange(1, len(history['train_loss'])+1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='train_loss')
    plt.plot(epochs, history['val_loss'],   label='val_loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f"{model_name} Loss"); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='train_acc')
    plt.plot(epochs, history['val_acc'],   label='val_acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title(f"{model_name} Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{model_name}_history.png"))
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, outpath=None):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-12)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', xticklabels=classes, yticklabels=classes, cmap=cmap)
    plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.title(title)
    if outpath:
        plt.savefig(outpath)
    plt.show()

# Visualize a batch
print("Showing sample training batch...")
show_batch(train_loader, class_names, nrow=4, title="Sample training images")

# ---------------------------
# Model definitions (custom)
# ---------------------------

# 1) Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #56
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), #28
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2) VGG-like small
class VGGLike(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(VGGLike, self).__init__()
        def conv_block(in_c, out_c, n=2):
            layers = []
            for i in range(n):
                layers += [nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_c),
                           nn.ReLU(inplace=True)]
                in_c = out_c
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            return nn.Sequential(*layers)
        self.features = nn.Sequential(
            conv_block(3, 32, n=2),   # 112
            conv_block(32, 64, n=2),  # 56
            conv_block(64, 128, n=2), # 28
            conv_block(128, 256, n=2) # 14
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 3) ResNet-like small (BasicBlock)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetLike(nn.Module):
    def __init__(self, block=BasicBlock, layers=[2,2,2], num_classes=NUM_CLASSES):
        super(ResNetLike, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # ->112
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # ->56
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# 4) Simple DenseNet-like (not full DenseNet - a small "dense block" style)
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32, n_layers=3):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(n_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False)
            ))
            channels += growth_rate
        self.out_channels = channels
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features,1)

class SimpleDenseNetLike(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(SimpleDenseNetLike, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) #112
        self.pool = nn.MaxPool2d(3, stride=2, padding=1) #56
        self.db1 = DenseBlock(64, growth_rate=32, n_layers=3)
        self.trans1 = nn.Sequential(nn.BatchNorm2d(self.db1.out_channels), nn.ReLU(inplace=True),
                                    nn.Conv2d(self.db1.out_channels, 128, kernel_size=1), nn.AvgPool2d(2))
        self.db2 = DenseBlock(128, growth_rate=32, n_layers=3)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.db2.out_channels, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.db1(x)
        x = self.trans1(x)
        x = self.db2(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

# 5) MobileNet-like light (depthwise separable convs)
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNetLite(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(MobileNetLite, self).__init__()
        self.model = nn.Sequential(
            conv_bn(3, 32, 2), #112
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            *[conv_dw(512,512,1) for _ in range(2)],
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.model(x)

# ---------------------------
# Training & evaluation utilities
# ---------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc

def eval_model(model, dataloader, criterion, device, return_preds=False):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_outputs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
            all_outputs.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    if return_preds:
        return epoch_loss, epoch_acc, torch.cat(all_outputs), torch.cat(all_preds), torch.cat(all_labels)
    else:
        return epoch_loss, epoch_acc

# ---------------------------
# Train multiple models
# ---------------------------
models_to_train = {
    "SimpleCNN": SimpleCNN(num_classes=len(class_names)),
    "VGGLike": VGGLike(num_classes=len(class_names)),
    "ResNetLike": ResNetLike(num_classes=len(class_names)),
    "SimpleDenseNetLike": SimpleDenseNetLike(num_classes=len(class_names)),
    "MobileNetLite": MobileNetLite(num_classes=len(class_names)),
}

trained_models = {}

for model_name, model in models_to_train.items():
    print("\n" + "="*80)
    print(f"Training model: {model_name}")
    print("="*80)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_LR_STEP, gamma=STEP_LR_GAMMA)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = defaultdict(list)

    for epoch in range(1, NUM_EPOCHS+1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = eval_model(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_best.pt"))

        t1 = time.time()
        print(f"Epoch {epoch}/{NUM_EPOCHS} | train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} | time: {(t1-t0):.1f}s")

    # Load best weights
    model.load_state_dict(best_model_wts)
    # Save final model too
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{model_name}_final.pt"))

    # Save history CSV and plot
    history_path = os.path.join(OUTPUT_DIR, f"{model_name}_history.csv")
    save_history_csv(history, history_path)
    plot_history(history, model_name, OUTPUT_DIR)

    trained_models[model_name] = model
    # also save a small metadata file
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_meta.json"), "w") as f:
        json.dump({"model_name": model_name, "best_val_acc": best_acc, "epochs": NUM_EPOCHS}, f, indent=2)

# ---------------------------
# Evaluate models on test set & save metrics
# ---------------------------
def evaluate_and_report(model, loader, device, classes):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    cm = confusion_matrix(all_labels, all_preds)
    cls_report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    p_r_f = precision_recall_fscore_support(all_labels, all_preds, average=None, labels=range(len(classes)))
    accuracy = (all_preds == all_labels).mean()
    return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": cls_report, "prf": p_r_f, "preds": all_preds, "labels": all_labels, "probs": all_probs}

overall_results = {}
for model_name, model in trained_models.items():
    print("\n" + "-"*60)
    print("Testing model:", model_name)
    print("-"*60)
    res = evaluate_and_report(model.to(DEVICE), test_loader, DEVICE, class_names)
    overall_results[model_name] = res
    # Save classification report
    with open(os.path.join(OUTPUT_DIR, f"{model_name}_classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {res['accuracy']:.4f}\n\n")
        f.write(res['classification_report'])
    # Save confusion matrices plots
    plot_confusion_matrix(res['confusion_matrix'], class_names, normalize=False,
                          title=f"{model_name} Confusion Matrix", outpath=os.path.join(OUTPUT_DIR, f"{model_name}_cm.png"))
    plot_confusion_matrix(res['confusion_matrix'], class_names, normalize=True,
                          title=f"{model_name} Confusion Matrix (Normalized)", outpath=os.path.join(OUTPUT_DIR, f"{model_name}_cm_norm.png"))

    # Save predictions csv
    csv_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true_label", "pred_label"] + [f"prob_{c}" for c in class_names])
        for i, (t, p, prob) in enumerate(zip(res['labels'], res['preds'], res['probs'])):
            writer.writerow([i, class_names[t], class_names[p]] + list(map(float, prob)))

    print(f"Test Accuracy: {res['accuracy']:.4f}")
    print("Classification report:\n", res['classification_report'])

# ---------------------------
# Visualize some sample predictions for the best model (by test accuracy)
# ---------------------------
best_model_name = max(overall_results.items(), key=lambda x: x[1]['accuracy'])[0]
print(f"\nBest model on test set: {best_model_name} (accuracy={overall_results[best_model_name]['accuracy']:.4f})")
best_model = trained_models[best_model_name]
best_model.to(DEVICE)

# Show some test images with predictions
def show_predictions(model, loader, classes, device, num_images=12):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(14,10))
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            inputs = inputs.cpu()
            for i in range(inputs.size(0)):
                if images_shown >= num_images:
                    break
                ax = plt.subplot(3, 4, images_shown+1)
                imshow_tensor(inputs[i], mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                true_lbl = classes[labels[i].item()]
                pred_lbl = classes[preds[i].item()]
                ax.set_title(f"T:{true_lbl}\nP:{pred_lbl}\n{probs[i].max().item():.2f}")
                images_shown += 1
            if images_shown >= num_images:
                break
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{best_model_name}_sample_predictions.png"))
    plt.show()

show_predictions(best_model, test_loader, class_names, DEVICE, num_images=12)

# Save overall summary
summary = {}
for mname, res in overall_results.items():
    summary[mname] = {"test_accuracy": float(res['accuracy'])}
with open(os.path.join(OUTPUT_DIR, "summary_results.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("All done. Results saved to:", OUTPUT_DIR)
