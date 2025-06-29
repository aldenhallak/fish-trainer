import os
import torch
from train_fish_doodle_classifier import get_model, DEVICE, basic_transform
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report

MODEL_PATH = "fish_doodle_classifier.pth"
DATASET_DIR = "dataset"

# Load model
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def predict(model, image_path):
    image = Image.open(image_path)
    image = basic_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()
    return prob

# Prepare test data
fish_dir = os.path.join(DATASET_DIR, "fish")
not_fish_dir = os.path.join(DATASET_DIR, "not_fish")

fish_imgs = [os.path.join(fish_dir, f) for f in os.listdir(fish_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
not_fish_imgs = [os.path.join(not_fish_dir, f) for f in os.listdir(not_fish_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

if __name__ == "__main__":
    model = load_model()
    y_true = []
    y_pred = []
    fish_probs = []
    not_fish_probs = []
    for img_path in tqdm(fish_imgs, desc="Testing fish"):
        prob = predict(model, img_path)
        y_true.append(1)  # fish is now positive class (1)
        y_pred.append(int(prob > 0.5))  # predict fish if prob > 0.5
        fish_probs.append(prob)
    for img_path in tqdm(not_fish_imgs, desc="Testing not_fish"):
        prob = predict(model, img_path)
        y_true.append(0)  # not_fish is negative class (0)
        y_pred.append(int(prob > 0.5))  # predict fish if prob > 0.5
        not_fish_probs.append(prob)
    print(classification_report(y_true, y_pred, target_names=["not_fish", "fish"]))
    print("\nSample fish probabilities:", fish_probs[:10])
    print("Sample not_fish probabilities:", not_fish_probs[:10])

    # Test on external images
    for test_img, label in [("fish.png", "fish"), ("not_fish.png", "not_fish")]:
        if os.path.exists(test_img):
            prob = predict(model, test_img)
            print(f"{test_img}: Probability of fish = {prob:.4f} => Predicted: {'fish' if prob > 0.5 else 'not_fish'}")
        else:
            print(f"{test_img} not found.")
