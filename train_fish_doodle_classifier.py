import os
import json
import argparse
import urllib.request
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# === CONFIG ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4  # Lower learning rate for more stable training
VAL_SPLIT = 0.2
WEIGHT_DECAY = 5e-4  # Stronger L2 regularization

# === UTILS: PIL loader with white background ===
def pil_loader_white_bg(path):
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and 'transparency' in img.info):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    else:
        img = img.convert("RGB")
    return img

# === TRANSFORMS ===
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: pil_loader_white_bg(img) if isinstance(img, str) else img),
    # Convert to grayscale for doodle classification - color is just noise
    transforms.Grayscale(num_output_channels=3),  # Repeat grayscale to 3 channels for pretrained model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats still work
])

# === MODEL ===

def get_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    
    return model.to(DEVICE)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, input, target):
        bce_loss = self.bce(input, target)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

# === TRAINING FUNCTION ===
def train(model, train_loader, val_loader, class_weights, epochs=5, patience=5):
    pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(DEVICE)
    print(f"Using pos_weight={pos_weight.item():.3f} for BCEWithLogitsLoss")

    # You can swap this out with FocalLoss(alpha=0.25, gamma=2.0) if needed
    criterion = FocalLoss(alpha=0.8, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=8)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f} | Train Acc: {acc:.4f}")
        val_loss = evaluate(model, val_loader, return_loss=True)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            print(f"No improvement in val loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

# === THRESHOLD OPTIMIZATION ===
def find_optimal_threshold(model, loader):
    """Find optimal threshold for classification based on F1 score"""
    model.eval()
    y_true = []
    y_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_true.extend(labels.numpy())
            y_probs.extend(probs)

    from sklearn.metrics import f1_score, recall_score
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (np.array(y_probs) > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, average='macro')
        recall_fish = recall_score(y_true, y_pred, pos_label=0)
        recall_not_fish = recall_score(y_true, y_pred, pos_label=1)
        balanced_recall = (recall_fish + recall_not_fish) / 2
        combined_score = 0.7 * f1 + 0.3 * balanced_recall

        if combined_score > best_score:
            best_score = combined_score
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.3f} (Combined Score: {best_score:.3f})")
    plot_roc(y_true, y_probs)  # You'll need to pass these in from `find_optimal_threshold`
    return best_threshold

# === EVALUATION ===
def evaluate(model, loader, threshold=0.5, return_loss=False):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > threshold).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            if return_loss:
                labels = labels.to(DEVICE).float().unsqueeze(1)
                total_loss += criterion(outputs, labels).item()

    from sklearn.metrics import confusion_matrix
    print(classification_report(y_true, y_pred, target_names=['fish', 'not_fish']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if return_loss:
        return total_loss


# === DATASET SPLIT LOADER ===
def load_dataset(data_dir):
    
    # Check original class counts
    fish_count = len([f for f in os.listdir(os.path.join(data_dir, 'fish')) if f.endswith(('.png', '.jpg', '.jpeg'))])
    not_fish_count = len([f for f in os.listdir(os.path.join(data_dir, 'not_fish')) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Original counts - Fish: {fish_count}, Not-fish: {not_fish_count}")
        
    transform = basic_transform
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    labels = [label for _, label in full_dataset]

    train_idx, val_idx = train_test_split(
        np.arange(len(full_dataset)),
        test_size=VAL_SPLIT,
        stratify=labels,
        random_state=42
    )

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    label_counter = Counter([labels[i] for i in train_idx])
    print(f"Training class distribution: {dict(label_counter)}")
    
    # Balanced weighting - let's try equal importance first
    weights = [1.0 / label_counter[labels[i]] for i in train_idx]
    
    sampler = WeightedRandomSampler(weights, len(weights))

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, label_counter

# === MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset', help='Path to dataset dir')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain with QuickDraw')
    args = parser.parse_args()
    
    print("Loading dataset...")
    train_loader, val_loader, class_weights = load_dataset(args.data)

    print("Initializing model...")
    model = get_model()

    print("Training...")
    train(model, train_loader, val_loader, class_weights, epochs=EPOCHS)

    print("Finding optimal threshold...")
    optimal_threshold = find_optimal_threshold(model, val_loader)
    
    print(f"Final evaluation with threshold {optimal_threshold:.3f}:")
    evaluate(model, val_loader, threshold=optimal_threshold)

    print("Exporting model to fish_doodle_classifier.onnx...")
    model.eval()  # Ensure eval mode before export
    dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        "fish_doodle_classifier.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print("Saving model to fish_doodle_classifier.pth...")
    torch.save(model.state_dict(), "fish_doodle_classifier.pth")
    
    # Save optimal threshold for later use
    with open("optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))
    print(f"Optimal threshold {optimal_threshold:.3f} saved to optimal_threshold.txt")
    


if __name__ == "__main__":
    main()
