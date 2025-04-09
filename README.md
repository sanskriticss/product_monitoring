# 🛠️ Product Monitoring using YOLOv8

This project implements a **real-time product monitoring system** using machine learning to **count objects** on a conveyor belt and **detect defects**. Built with **YOLOv8 segmentation models**, it ensures quality control by detecting, tracking, and analyzing products as they move along the belt.

Google drive for videos and dataset:
https://drive.google.com/drive/folders/1myRSzQj-QJPx75mNUxolewzxlyu_ikZ4?usp=sharing

---

## Features

- **Object Detection:** Detects various products using YOLO segmentation.
- **Object Tracking:** Maintains unique object identities with a custom tracker.
- **Defect Detection:** Detects defective products using learned quality standards.
- **Object Counting:** Counts objects within a defined region of interest (ROI).

---

## Dependencies

Install the required packages via pip:
pip install opencv-python numpy yolov5 cvzone ultralytics

## Model & Dataset

**Model:** YOLOv8 segmentation model (fine-tuned from `yolov8n.pt`)

**Dataset:**
- ~5,000 images per product category (~40,000 images total)
- Sourced from real-world conveyor belt footage and public datasets
- Augmented for robustness

**Annotations:** Manual labeling with Roboflow (bounding boxes + segmentation masks)  
**Accuracy:** 97.6% on the validation dataset

---

## Training the Model

**Base Model:** YOLOv8 Nano (`yolov8n.pt`)  
**Training Time:** ~5 hours per product category

**Training Process:**
- Fine-tuned on custom-labeled images
- Hyperparameters (learning rate, batch size, augmentations) adjusted for optimal results

**Tuning Methods:**
- Confidence and IoU thresholds adjusted
- Anchor sizes optimized
- Early stopping used to prevent overfitting

**Hardware Recommendation:** NVIDIA RTX 3090 or A100 for best performance

---

## Code Structure

### 1. Initialization
- Loads YOLO segmentation models for each product type
- Reads class labels from text file
- Captures video feed for processing

### 2. Defining ROI & Tracking
- Defines a Region of Interest (ROI) on the conveyor belt
- Assigns unique IDs to detected objects using a tracking class

### 3. Processing Video Frames
- Resizes and reads each frame
- Detects objects using YOLO
- Updates tracker with bounding boxes
- Checks if objects enter the ROI
- Displays tracking output on screen

### 4. Object Counting & Display
- Counts objects only if they remain in the ROI for a defined number of frames
- Displays the object count using `cvzone.putTextRect`

---

## Usage

Ensure that the models and video files are present in the working directory. To run the system:

---

## Conclusion

This system offers a robust ML-based solution for real-time product monitoring and defect detection on conveyor belts. It helps automate quality control processes, improving reliability and reducing manual effort in industrial environments.
