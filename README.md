# AIDL24_SelfDriving

# Table of contents
* [Motivation/Hypothesis](#1-motivation)
* [Dataset](#dataset)
	* [Anlaysis of the Dataset](#analysis-of-the-datset)
* [Models Used](#models)
  * [LaneNet](#LaneNet)
  * [MaskRCNN](#MaskRCNN)
  * [FasterRCNN](#FasterRCNN)
* [Loss Function](#loss-functions)
* [Evaluation Metrics](#evaluation-metrics)
     * [Intersection Over Union](#intersection-over-union-iou)
     * [Mean Average Precision](#mean-average-precision-map)
     * ---
* [Computational Resources](#computational-resources)
* [How To Run](#how-to-run)
* [Training Models](#training-Models)
     * [Challenges](#challenges)
     * [Exploiding Gradients](#exploding-gradients)
     * [Predictions Positionins](#predictions-position)
* [Transfer Learning](#transfer-learning) ???
     * [Introduction To Transfer Learning](#introduction-to-transfer-learning)
     * [Application of Transfer Learning](#application-of-transfer-learning-in-this-project)
     * [Transfer Learning Code](#transfer-learning-code)
* [Models Comparision](#models-comparison)
* [Validation With Our Own Images](#validation-with-our-own-images)
* [Conclusion And Future Work](#conclusion-and-future-work)
* [References](#references)

Certainly. I'll revise the README content to match the table of contents you've provided:

```markdown
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

We utilize [Dataset Name], which contains [X] images of diverse road scenes. This dataset is crucial for our project due to its comprehensive coverage of various driving conditions and high-quality annotations.

## Analysis of the Dataset

Our dataset comprises [X] training images and [Y] validation images. We've conducted thorough preprocessing, including normalization and augmentation, to optimize model performance.

## Models

### LaneNet

LaneNet is our primary model for lane detection. It employs a segmentation-based approach to identify and classify lane markings accurately.

### MaskRCNN

We use MaskRCNN for instance segmentation, allowing us to precisely identify and locate objects in the driving scene.

### FasterRCNN

FasterRCNN is our chosen model for object detection, providing rapid and accurate identification of various objects on the road.

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
