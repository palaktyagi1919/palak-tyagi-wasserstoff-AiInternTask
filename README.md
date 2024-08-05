# palak-tyagi-wasserstoff-AiInternTask
AI pipeline for image segmentation and object analysis
## Overview
The goal is to create a pipeline using transformers or deep learning models to process an input image, segment and identify objects within the image, and produce a summary table with mapped data for each object.The pipeline leverages various deep learning models and image processing libraries to achieve these tasks.

## Features
*Image Segmentation:* Uses Mask R-CNN to segment objects in images. <br>
*Object Identification:* Identifies objects using Faster R-CNN.<br>
*Text Extraction:* Extracts text from segmented objects using Tesseract OCR.<br>
*Text Summarization:* Summarizes extracted text using NLP models.<br>
*Interactive UI:* Provides an interactive Streamlit web interface for image upload and processing.<br>
*Data Mapping:* Maps object data and attributes to the original image.<br>


## Project Structure
.
├── app.py                  # Streamlit app script<br>
├── README.md               # Project README file<br>
├── requirements.txt        # List of required Python packages<br>
├── data/                   # Directory for storing input and output data<br>
│   ├── input_images/       # Directory for input images<br>
│   ├── segmented_objects/  # Directory for segmented object images<br>
│   └── output/             # Directory for output data (JSON, CSV)<br>
└── scripts/                # Directory for additional scripts (if any)<br>

## Workflow
### 1. Image Segmentation:
* Uses Mask R-CNN to segment objects within an image.
* Saves segmented objects as separate images.
* 
### Step 2: Object Extraction and Storage
*Task*: Extract each segmented object from the image and store separately with unique IDs.

*Deliverables*:
- Code to extract each segmented object and save them as separate images.
- Assign a unique ID for each object and a master ID for the original image.
- Save the object images and their metadata (unique ID, master ID) in a file system or database.

*Tools/Resources*: OpenCV, PIL, SQLite or any preferred database.

### Step 3: Object Identification
*Task*: Identify each object and describe what they are in the real world.

*Deliverables*:
- Implement a model or use a pre-trained model (e.g., YOLO, Faster R-CNN, CLIP) to identify and describe objects.
- Code to generate a description for each object image.
- Document containing the identified objects and their descriptions.

*Suggested Tools/Resources*: Pre-trained object detection models, CLIP.

### Step 4: Text/Data Extraction from Objects
*Task*: Extract text or data from each object image.

*Deliverables*:
- Implement or use a pre-trained model (e.g., Tesseract OCR, EasyOCR) for text extraction.
- Code to extract and store text/data from each object image.
- Document containing extracted text/data for each object.

*Suggested Tools/Resources*: OCR tools, PyTorch, TensorFlow.

### Step 5: Summarize Object Attributes
*Task*: Summarize the nature and attributes of each object.

*Deliverables*:
- Code to generate a summary of the nature and attributes of each object.
- Document containing summarized attributes for each object.

*Suggested Tools/Resources*: NLP models, summarization algorithms.

### Step 6: Data Mapping
*Task*: Map all extracted data and attributes to each object and the master input image.

*Deliverables*:
- Code to map unique IDs, descriptions, extracted text/data, and summaries to each object.
- Data structure (e.g., JSON, database schema) representing the mapping.

*Suggested Tools/Resources*: JSON, SQL, any preferred database.

### Step 7: Output Generation
*Task*: Output the original image along with a table containing all mapped data for each object in the master image.

*Deliverables*:
- Code to generate the final output image with annotations.
- Table summarizing all data mapped to each object and the master image.
- Final visual output showing the original image with segmented objects and an accompanying table.
## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

1. Fork the repository.
2. Create your feature branch (git checkout -b feature/your-feature-name).
3. Commit your changes (git commit -m 'Add some feature').
4. Push to the branch (git push origin feature/your-feature-name).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
