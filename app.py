import streamlit as st
from PIL import Image
import torchvision.transforms as T
import torch
import torchvision.models as models
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the function to visualize predictions
def visualize_predictions(image, predictions, threshold=0.5):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for i, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            plt.imshow(mask, alpha=0.5)
    plt.axis('off')
    st.pyplot(plt)

# Define the function to save segmented objects
def save_segmented_objects(image, predictions, output_dir="segmented_objects", threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            masked_image = np.array(image) * mask[:, :, np.newaxis]
            object_image = Image.fromarray(masked_image.astype(np.uint8))
            object_image.save(os.path.join(output_dir, f"object_{i}.png"))
            st.write(f"Saved object_{i}.png")

# Streamlit application
st.title("Image Segmentation with Mask R-CNN")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Transform the image
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image)

    # Load the model
    model = models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
    model.eval()

    # Check if GPU is available and use it if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predict
    with torch.no_grad():
        predictions = model([image_tensor.to(device)])

    # Visualize and save segmented objects
    visualize_predictions(image, predictions)
    save_segmented_objects(image, predictions)
