#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ! pip install torch torchvision tensorflow pillow opencv-python
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:


from google.colab import drive
drive.mount('/content/drive/')


# In[ ]:


from PIL import Image
import torchvision.transforms as T


# In[ ]:


image = Image.open("/content/drive/MyDrive/file 2.jpg")
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image)


# In[ ]:


import torchvision.models as models
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Load a Mask R-CNN model with specific weights
model = models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
model.eval()
# Check if GPU is available and use it if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[ ]:


with torch.no_grad():
    predictions = model([image_tensor])


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(image, predictions, threshold=0.5):
    plt.imshow(image)
    for i, mask in enumerate(predictions[0]['masks']):
        if predictions[0]['scores'][i] > threshold:
            mask = mask[0].mul(255).byte().cpu().numpy()
            plt.imshow(mask, alpha=0.5)
    plt.show()


# In[ ]:


visualize_predictions(image, predictions)


# visualize_predictions(image, predictions)

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


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


# In[ ]:


save_segmented_objects(image, predictions)


# save_segmented_objects(image, predictions)

# In[ ]:


get_ipython().system(' pip install transformers')


# In[ ]:


from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# In[ ]:


def describe_objects(output_dir="segmented_objects"):
    descriptions = {}
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(output_dir, filename)
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_text_features(**inputs)
            text = model.generate(inputs)
            descriptions[filename] = text

    return descriptions
    object_descriptions = describe_objects()



# In[ ]:


pip install pytesseract


# In[ ]:


import pytesseract


# In[ ]:


get_ipython().system('sudo apt-get install tesseract-ocr')


# In[ ]:


def extract_text_from_objects(output_dir="segmented_objects"):
    texts = {}
    for filename in os.listdir(output_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(output_dir, filename)
            text = pytesseract.image_to_string(image_path)
            texts[filename] = text

    return texts
object_texts = extract_text_from_objects()


# In[ ]:


from transformers import pipeline

summarizer = pipeline("summarization")

def summarize_object_attributes(object_texts):
    summaries = {}
    for filename, text in object_texts.items():
        summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
        summaries[filename] = summary[0]['summary_text']

    return summaries

object_summaries = summarize_object_attributes(object_texts)


# In[ ]:


import json
import os

def create_data_mapping(object_descriptions, object_texts, object_summaries, output_dir="segmented_objects"):
    data_mapping = {}

    try:
        for filename in os.listdir(output_dir):
            if filename.endswith(".png"):
                object_id = filename.split(".")[0]
                data_mapping[object_id] = {
                    "description": object_descriptions.get(filename, ""),
                    "text": object_texts.get(filename, ""),
                    "summary": object_summaries.get(filename, "")
                }

        with open("data_mapping.json", "w") as f:
            json.dump(data_mapping, f, indent=4)
    except FileNotFoundError as e:
        print(f"Error: Directory not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage
object_descriptions = {"object1.png": "description1", "object2.png": "description2"}
object_texts = {"object1.png": "text1", "object2.png": "text2"}
object_summaries = {"object1.png": "summary1", "object2.png": "summary2"}

create_data_mapping(object_descriptions, object_texts, object_summaries)


# In[ ]:





# In[ ]:


import pandas as pd

def generate_final_output(image, data_mapping):
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)
    table_data = []

    for object_id, data in data_mapping.items():
        table_data.append([object_id, data["description"], data["text"], data["summary"]])
        # Here you would add code to annotate the image with object IDs

    table = pd.DataFrame(table_data, columns=["Object ID", "Description", "Text", "Summary"])
    table.to_csv("object_data.csv", index=False)

    plt.show()

with open("data_mapping.json") as f:
    data_mapping = json.load(f)

generate_final_output(image, data_mapping)


# In[ ]:


get_ipython().system('pip install streamlit')


# In[ ]:


pip install opencv-python-headless pillow pandas


# In[ ]:


import streamlit as st
from PIL import Image
import pandas as pd
import cv2
import json
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)


# In[ ]:


st.title("AI Pipeline for Image Segmentation and Object Analysis")

uploaded_file = st.file_uploader("World_map", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Segmenting image...")


# In[ ]:


st.title("AI Pipeline for Image Segmentation and Object Analysis")

uploaded_file = st.file_uploader("/content/drive/MyDrive/file 2.jpg", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Segmenting image...")


# In[ ]:


if uploaded_file is not None:
    # Dummy function for segmentation
    segmented_objects, master_id = segment_image(image)

    # Display segmented objects
    st.image(segmented_objects, caption='Segmented Objects', use_column_width=True)


# In[ ]:


if uploaded_file is not None:
    object_images = extract_objects(segmented_objects)

    for idx, obj in enumerate(object_images):
        st.image(obj, caption=f'Object {idx+1}', use_column_width=True)


# In[ ]:


if uploaded_file is not None:
    identified_objects = identify_objects(object_images)

    for idx, desc in enumerate(identified_objects):
        st.write(f"Object {idx+1}: {desc}")


# In[ ]:


if uploaded_file is not None:
    extracted_texts = extract_text(object_images)

    for idx, text in enumerate(extracted_texts):
        st.write(f"Object {idx+1} Text: {text}")


# In[ ]:


if uploaded_file is not None:
    summaries = summarize_attributes(identified_objects, extracted_texts)

    for idx, summary in enumerate(summaries):
        st.write(f"Object {idx+1} Summary: {summary}")


# In[ ]:


if uploaded_file is not None:
    # Create a DataFrame for the table
    data = {
        'Object ID': range(1, len(object_images) + 1),
        'Description': identified_objects,
        'Extracted Text': extracted_texts,
        'Summary': summaries
    }

    df = pd.DataFrame(data)
    st.write("Summary Table")
    st.dataframe(df)

    # Assuming you have a function to generate the final image with annotations
    annotated_image = generate_annotated_image(image, segmented_objects)
    st.image(annotated_image, caption='Annotated Image', use_column_width=True)


# In[ ]:


streamlit run app.py


# In[ ]:


streamlit run image_segmentation_pipeline.py

