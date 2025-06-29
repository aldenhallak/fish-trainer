import os
import torch
from train_fish_doodle_classifier import get_model, DEVICE, basic_transform, pil_loader_white_bg
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report
import onnxruntime as ort
import numpy as np

MODEL_PATH = "fish_doodle_classifier.pth"
ONNX_MODEL_PATH = "fish_doodle_classifier.onnx"
DATASET_DIR = "dataset"

# Load model
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# Load ONNX model
onnx_session = None
def load_onnx_model():
    global onnx_session
    if onnx_session is None:
        onnx_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    return onnx_session

def save_transformed_img(transformed, image_path):
    # Convert tensor to numpy, unnormalize, and save as white background
    img_np = (transformed * 0.5 + 0.5).clamp(0, 1).mul(255).byte().cpu().numpy()
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)  # (1, H, W) -> (H, W)
    img_pil = Image.fromarray(img_np.astype(np.uint8), mode="L")
    img_pil.save("transformed_img_" + os.path.basename(image_path))

def predict(model, image_path, save_transformed=False):
    image = pil_loader_white_bg(image_path)
    transformed = basic_transform(image)
    if save_transformed:
        save_transformed_img(transformed, image_path)
    # ONNX expects numpy array, shape (1,1,224,224), float32
    input_tensor = transformed.unsqueeze(0).cpu().numpy().astype(np.float32)
    session = load_onnx_model()
    outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
    prob = 1 / (1 + np.exp(-outputs[0].item()))  # sigmoid
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
    # Save transformed image for the first fish image
    if fish_imgs:
        predict(model, fish_imgs[0], save_transformed=True)
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
    for test_img, label in [("fish_test.png", "fish"), ("not_fish.png", "not_fish")]:
        if os.path.exists(test_img):
            prob = predict(model, test_img, save_transformed=True)
            print(f"{test_img}: Probability of fish = {prob:.4f} => Predicted: {'fish' if prob > 0.5 else 'not_fish'}")
        else:
            print(f"{test_img} not found.")
