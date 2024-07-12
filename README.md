# AIDL24_SelfDriving

## Table of contents
* [Motivation/Hypothesis](#1-motivation)
* [Dataset](#dataset)
  * [Analysis of the Dataset](#analysis-of-the-datset)
* [Models Used](#models)
  * [LaneNet](#lanenet)
  * [MaskRCNN](#maskrcnn)
  * [FasterRCNN](#fasterrcnn)
* [Loss Function](#loss-functions)
* [Evaluation Metrics](#evaluation-metrics)
  * [Intersection Over Union](#intersection-over-union-iou)
  * [Mean Average Precision](#mean-average-precision-map)
* [Computational Resources](#computational-resources)
* [How To Run](#how-to-run)
* [Training Models](#training-models)
* [Challenges](#challenges)
  * [Exploding Gradients](#exploding-gradients)
  * [Predictions Positioning](#predictions-position)
* [Transfer Learning](#transfer-learning)
  * [Introduction To Transfer Learning](#introduction-to-transfer-learning)
  * [Application of Transfer Learning](#application-of-transfer-learning-in-this-project)
  * [Transfer Learning Code](#transfer-learning-code)
* [Models Comparison](#models-comparison)
* [Validation With Our Own Images](#validation-with-our-own-images)
* [Conclusion And Future Work](#conclusion-and-future-work)
* [References](#references)

## 1. Motivation

Our project aims to advance the field of autonomous driving through improved computer vision techniques. We hypothesize that by integrating state-of-the-art models for lane detection and object recognition, we can enhance the accuracy and reliability of self-driving systems.

## Dataset

We use [BDD100K Dataset](https://www.vis.xyz/bdd100k/), which contains 100.000 images of diverse road scenes. This dataset is crucial for our project due to its comprehensive coverage of various driving conditions and high-quality annotations.\
You can download the dataset from [here](https://dl.cv.ethz.ch/bdd100k/data/)

### Analysis of the Dataset

The BDD100K dataset is a comprehensive collection of 100,000 images capturing diverse road scenes. This dataset is pivotal for projects involving autonomous driving, computer vision, and scene understanding due to its extensive coverage of various driving conditions and high-quality annotations.

#### 1. Overview

* **Number of Images**: 100,000
* **Resolution**: 1280x720 pixels
* **Annotations**: Multiple types including object detection, lane marking, drivable area, and segmentation.
* **Driving Conditions**: Diverse, covering different weather conditions, times of day, and various locations.

#### 2. Data Types and Annotations
* **Object Detection**: Bounding boxes for objects such as cars, pedestrians, traffic signs, and cyclists.
* **Lane Marking**: Polylines indicating the position of lane markings on the road.
* **Drivable Area**: Segmentation maps identifying areas that are drivable.
* **Semantic Segmentation**: Pixel-level labels for different objects and regions in the image.
* **Instance Segmentation**: Pixel-level labels with instance information for objects.

## Models

### LaneNet

LaneNet is our primary model for lane detection. It employs a segmentation-based approach to identify and classify lane markings accurately.

### MaskRCNN

This model is designed to perform lane detection using a combination of convolutional neural network (CNN) layers, feature pyramid networks (FPN), region of interest (RoI) alignment, and semantic segmentation heads.

## Components

#### CustomBackbone

This component is a convolutional neural network that extracts feature maps from input images. It uses:
- Several convolutional layers
- Batch normalization
- ReLU activations

The feature maps are extracted through a series of layers.

#### FeaturePyramidNetwork

The FPN creates feature pyramids from the backbone's output. It:
- Combines feature maps from different levels (layers)
- Builds multi-scale feature maps
- Useful for detecting objects at various scales

#### PyramidRoIAlign

This module aligns Regions of Interest (RoIs) of different sizes to a fixed size using feature maps from the FPN. It:
- Uses the spatial scale of the feature maps to properly resize and align the RoIs
- Distributes RoIs to different levels of the pyramid based on their size

#### SemanticLaneHead

This head performs semantic segmentation for lane detection. It consists of:
- Multiple convolutional layers
- A deconvolutional (upsampling) layer
- A final convolutional layer to produce the segmentation mask logits

#### LaneDetectionModel

This is the complete model that integrates all the components. It:
- Takes images and RoIs as input
- Processes them through the backbone, FPN, RoI align, and the semantic lane head
- Produces the final lane detection mask logits


### Faster R-CNN with ResNet50-FPN

## Faster R-CNN Architecture

Faster R-CNN is a two-stage object detection algorithm:

### Region Proposal Network (RPN)
- Scans the image and proposes potential object regions
- Uses anchor boxes of various sizes and aspect ratios
- Outputs "objectness" scores and rough bounding box coordinates

### Fast R-CNN
- Takes proposed regions from RPN
- Performs classification (what object is it?)
- Refines bounding box coordinates

## ResNet50 Backbone

ResNet50 is a deep convolutional neural network with 50 layers:

- Uses residual connections (skip connections)
- Allows training of very deep networks by addressing vanishing gradient problem
- Composed of repetitive blocks:
  - Convolutional layers
  - Batch normalization
  - ReLU activation functions

## Feature Pyramid Network (FPN)

FPN enhances feature extraction:

- Creates a multi-scale feature pyramid
- Top-down pathway: Upsamples spatially coarser, but semantically stronger features
- Lateral connections: Merges features from the bottom-up and top-down pathways
- Helps detect objects across a wide range of scales

## Pre-trained Weights (COCO_V1)

- COCO (Common Objects in Context) dataset:
  - 330K images
  - 1.5 million object instances
  - 80 object categories
- Pre-training on COCO provides a strong starting point for transfer learning

## Customization Options

- Modify number of classes for your specific task
- Adjust anchor sizes and aspect ratios
- Fine-tune learning rates and other hyperparameters
- Freeze/unfreeze different parts of the network during training

## Performance Considerations

- Inference speed vs accuracy trade-off
- GPU memory requirements
- Potential for model quantization or pruning for deployment on resource-constrained devices

## Loss Functions

We implement custom loss functions tailored to each model's architecture and task requirements.

## Evaluation Metrics

### Intersection Over Union (IoU)

IoU is used to measure the accuracy of our object detection and segmentation models by comparing predicted bounding boxes with ground truth.

### Mean Average Precision (mAP)

mAP helps us evaluate the overall performance of our models across different object classes and detection thresholds.

## Computational Resources

This project was developed using [hardware/software details]. We utilized [specific GPUs/cloud resources] for training our models.

## How To Run

1. Clone the repository:
   ```
   git clone https://github.com/jsabahu/AIDL24_SelfDriving.git
   ```
2. Create and activate a conda environment:
   ```
   conda create --name AIDL24_SelfDriving python=3.11.9
   conda activate AIDL24_SelfDriving
   ```
3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Training Models

[Provide details on how to train each model, including hyperparameters and training scripts]

## Challenges

### Exploding Gradients

We addressed the issue of exploding gradients by [describe your solution, e.g., gradient clipping, adjusting learning rates].

### Predictions Positioning

To improve prediction positioning, we [describe your approach, e.g., implemented post-processing techniques, fine-tuned model architectures].

## Transfer Learning

### Introduction To Transfer Learning

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

### Application of Transfer Learning in This Project

We applied transfer learning to [describe which models and how it was applied].

### Transfer Learning Code

[Provide code snippets or explanations of how transfer learning was implemented]

## Models Comparison

[Present a detailed comparison of the performance of LaneNet, MaskRCNN, and FasterRCNN, including metrics and analysis]

## Validation With Our Own Images

We tested our models on a set of custom images to validate real-world performance. [Describe the results and provide examples]

## Conclusion And Future Work

Our project demonstrates [key findings]. Future work could focus on [potential improvements or new directions].

## References

1. [Reference 1]
2. [Reference 2]
3. ...

## Contributors

- Jordi Sabatés
- Marc Ramon
- Marc Giné
- Fermin Gomila

Advisor: Daniel Fojo
```

## Project installation

- Clone repository
```
git clone https://github.com/jsabahu/AIDL24_SelfDriving.git
```

- Create a conda environment
```
 conda create --name AIDL24_SelfDriving python=3.11.9
```

- Activate conda environment
```
 conda activate AIDL24_SelfDriving
```

- Install requirements packages
```
pip install -r requirements.txt
```
