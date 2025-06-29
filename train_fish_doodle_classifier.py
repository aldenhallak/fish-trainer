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

# === CONFIG ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2

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
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

augmented_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda img: pil_loader_white_bg(img) if isinstance(img, str) else img),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === MODEL ===
def get_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1)
    )
    return model.to(DEVICE)

# === TRAINING FUNCTION ===
def train(model, train_loader, val_loader, class_weights, epochs=5, patience=5):
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights[0]/class_weights[1]]).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            labels = 1 - labels
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"No improvement in val loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

# === EVALUATION ===
def evaluate(model, loader, return_loss=False):
    model.eval()
    y_true = []
    y_pred = []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for inputs, labels in loader:
            labels = 1 - labels
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            if return_loss:
                labels = labels.to(DEVICE).float().unsqueeze(1)
                total_loss += criterion(outputs, labels).item()
    print(classification_report(y_true, y_pred, target_names=['fish', 'not_fish']))
    if return_loss:
        return total_loss

# === DOWNLOAD .NDJSON FILE ===
def download_ndjson(class_name, out_dir="quickdraw"):
    os.makedirs(out_dir, exist_ok=True)
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{class_name}.ndjson"
    dest = os.path.join(out_dir, f"{class_name}.ndjson")
    if not os.path.exists(dest):
        print(f"Downloading {class_name}.ndjson...")
        urllib.request.urlretrieve(url, dest)
    return dest

# === DRAW .NDJSON DOODLES ===
def draw_ndjson_to_png(ndjson_path, out_dir, max_count=1000, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    with open(ndjson_path, 'r') as f:
        saved = 0
        for i, line in enumerate(f):
            drawing = json.loads(line)['drawing']
            total_points = sum(len(stroke[0]) for stroke in drawing)
            if total_points < 150:
                continue
            img = np.ones((256, 256), dtype=np.uint8) * 255
            for stroke in drawing:
                for j in range(len(stroke[0]) - 1):
                    pt1 = (int(stroke[0][j]), int(stroke[1][j]))
                    pt2 = (int(stroke[0][j+1]), int(stroke[1][j+1]))
                    cv2.line(img, pt1, pt2, 0, 2)
            # Only save if enough ink (non-white pixels)
            ink_pixels = np.sum(img < 250)  # count pixels that are not white
            if ink_pixels < 500:  # require at least 100 ink pixels
                continue
            filename = os.path.join(out_dir, f"{prefix}{saved}.png")
            cv2.imwrite(filename, img)
            saved += 1
            if saved >= max_count:
                break

# === MULTI-CLASS NOT-FISH BUILDER ===
def build_not_fish_from_classes(class_names, out_dir, max_per_class=1000):
    for cls in class_names:
        ndjson = download_ndjson(cls)
        draw_ndjson_to_png(ndjson, out_dir, max_count=max_per_class, prefix=f"{cls}_")
    print(f"Created not_fish set from {len(class_names)} classes.")

# === DATASET SPLIT LOADER ===
def load_dataset(data_dir, use_augmented=True):
    transform = augmented_transform if use_augmented else basic_transform
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
    weights = [1. / label_counter[labels[i]] for i in train_idx]
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

    if args.pretrain:
        print("🔧 Generating QuickDraw dataset...")

        # Fish
        fish_ndjson = download_ndjson("fish")
        draw_ndjson_to_png(fish_ndjson, os.path.join(args.data, "fish"), max_count=1950, prefix="fish_ndjson_")

        # Not fish: using multiple unrelated classes
        not_fish_classes = [
            "cat", "banana", "submarine", "face", "octopus",
            "crab", "broccoli", "cloud", "truck", "basket"
        ]
        build_not_fish_from_classes(not_fish_classes, os.path.join(args.data, "not_fish"), max_per_class=200)

    print("Loading dataset...")
    train_loader, val_loader, class_weights = load_dataset(args.data)

    print("Initializing model...")
    model = get_model()

    print("Training...")
    train(model, train_loader, val_loader, class_weights, epochs=EPOCHS)

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


if __name__ == "__main__":
    main()
