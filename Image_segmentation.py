
from PIL import Image
import torchvision.transforms as T

image = Image.open("C:\Users\Palak Tyagi\Downloads\Trafic_Dhaka.jpg""C:\Users\Palak Tyagi\Downloads\Trafic_Dhaka.jpg")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)

import torch

import torchvision.models as models
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Load a Mask R-CNN model with specific weights
model = models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()
# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    predictions = model([image_tensor])

import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(image, predictions, threshold=0.5):
    plt.imshow(image)
    for i, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            plt.imshow(mask, alpha=0.5)
    plt.show()

visualize_predictions(image, predictions)
import os

def save_segmented_objects(image, predictions, output_dir="segmented_objects", threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            masked_image = np.array(image) * mask[:, :, np.newaxis]
            object_image = Image.fromarray(masked_image.astype(np.uint8))
            object_image.save(os.path.join(output_dir, f"object_{i}.png"))
            print(f"Saved object_{i}.png")

save_segmented_objects(image, predictions)