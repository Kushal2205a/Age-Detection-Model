# Age Detection Model

![Python](https://img.shields.io/badge/python-3.7+-blue.svg) ![Dependencies](https://img.shields.io/badge/dependencies-auto--install-brightgreen) ![Status](https://img.shields.io/badge/status-active-success.svg)

A computer vision project that detects human faces in an image and predicts the age range. This implementation uses pre-trained Caffe models for face detection and age classification, leveraging OpenCV's Deep Neural Network (DNN) module.

---

## Demonstration

The model processes an input image to identify faces and overlays the predicted age bracket.

| Original Image | Processed Image |
| :---: | :---: |
| ([Test Picture](https://raw.githubusercontent.com/Kushal2205a/Age-Detection-Model/refs/heads/main/test_images/profile.webp)))  | ([Processed Picture](https://raw.githubusercontent.com/Kushal2205a/Age-Detection-Model/refs/heads/main/results/processed_profile.webp))) |

---

## About The Project

This project provides a straightforward way to perform age estimation from images. It uses a pre-trained Single Shot-Multibox Detector (SSD) model based on ResNet-10 for robust face detection and a separate Caffe model for classifying the detected face into one of eight predefined age groups.

The age brackets are: `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`.

---

## Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

* Python 3.7 or later
* pip package manager

### Installation

1.  Clone the repository or download the source code.
2.  The script `age_detector.py` is designed to automatically install the required Python packages (like `opencv-python` and `numpy`) if they are not found in your environment.
3.  Download the necessary Caffe models for face and age detection and place them in a `models/` directory in the project's root folder. The required files are:
    * `deploy.prototxt`
    * `res10_300x300_ssd_iter_140000.caffemodel`
    * `age_deploy.prototxt`
    * `age_net.caffemodel`

### Usage

1.  Create a directory named `test_images/` in the project's root.
2.  Place the images you want to process inside the `test_images/` directory.
3.  Run the script from your terminal:
    ```sh
    python age_detector.py
    ```
4.  The processed images with detected faces and age labels will be saved in a `results/` directory.

---

## How It Works

The age detection process is completed in two main stages:

1.  **Face Detection**: The script first loads the input image and creates a 300x300 blob from it. This blob is passed through a pre-trained SSD face detection network to locate faces within the image. A confidence threshold is used to filter out weak detections.

2.  **Age Prediction**: For each detected face, the Region of Interest (ROI) is extracted. This ROI is then converted into a blob and passed through the age prediction network. The network outputs a probability distribution over the predefined age buckets, and the bucket with the highest confidence is chosen as the predicted age range.

Finally, the script draws a bounding box around each detected face and labels it with the predicted age range before saving the new image.

