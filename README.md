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

LaneNet is our primary model for lane detection. It is a convolutional neural network (CNN) model designed specifically for lane detection in autonomous driving applications, which is crucial for tasks such as lane keeping, lane changing, and overall vehicle navigation.It employs a segmentation-based approach to identify and classify lane markings accurately.

#### Architecture
##### 1. Encoder-Decoder Structure:
LaneNet employs an encoder-decoder architecture to process input images and generate lane detection results. The encoder extracts high-level features from the input image using a series of convolutional layers, while the decoder reconstructs the spatial details to produce pixel-wise lane markings.

##### 2. Instance Segmentation:
LaneNet uses a semantic segmentation approach combined with instance segmentation. The model segments the image into lane and non-lane regions and differentiates between individual lane markings. This dual approach allows LaneNet to handle multiple lanes simultaneously and distinguish between them.

##### 3. Embedding Learning:
A key feature of LaneNet is its use of embedding vectors. For each pixel classified as a lane, the network assigns an embedding vector that helps cluster pixels belonging to the same lane. This clustering is achieved through a discriminative loss function that encourages pixels of the same lane to have similar embeddings while separating different lanes.

### MaskRCNN

This model is designed to perform lane detection using a combination of convolutional neural network (CNN) layers, feature pyramid networks (FPN), region of interest (RoI) alignment, and semantic segmentation heads.

#### Components

##### CustomBackbone

This component is a convolutional neural network that extracts feature maps from input images. It uses:
- Several convolutional layers
- Batch normalization
- ReLU activations

The feature maps are extracted through a series of layers.

##### FeaturePyramidNetwork

The FPN creates feature pyramids from the backbone's output. It:
- Combines feature maps from different levels (layers)
- Builds multi-scale feature maps
- Useful for detecting objects at various scales

##### PyramidRoIAlign

This module aligns Regions of Interest (RoIs) of different sizes to a fixed size using feature maps from the FPN. It:
- Uses the spatial scale of the feature maps to properly resize and align the RoIs
- Distributes RoIs to different levels of the pyramid based on their size

##### SemanticLaneHead

This head performs semantic segmentation for lane detection. It consists of:
- Multiple convolutional layers
- A deconvolutional (upsampling) layer
- A final convolutional layer to produce the segmentation mask logits

##### LaneDetectionModel

This is the complete model that integrates all the components. It:
- Takes images and RoIs as input
- Processes them through the backbone, FPN, RoI align, and the semantic lane head
- Produces the final lane detection mask logits

## Loss Function

The model uses Binary Cross-Entropy with Logits as the loss function for both training and evaluation:

```python
F.binary_cross_entropy_with_logits(output, masks)
```
This loss function combines a Sigmoid layer and the Binary Cross-Entropy loss in a single function. It's numerically stable and especially useful for tasks like lane detection where the output is a binary mask.
Key Features:

Automatically applies the sigmoid activation function to the model output before calculating the loss.
Provides better numerical stability than using a plain Sigmoid followed by Binary Cross-Entropy loss.

## Evaluation Metric
The primary evaluation metric used is Binary Accuracy:
```python
acc = binary_accuracy(output, masks, threshold=0.5)
```
Binary Accuracy
This metric calculates the accuracy of binary predictions:

It applies a threshold (default 0.5) to the model's output to create binary predictions.
It then compares these binary predictions to the ground truth masks.
The result is the proportion of correct predictions (both true positives and true negatives) among the total number of cases examined.

Key Points:

Threshold: A value of 0.5 is used to convert the model's probabilistic output into binary predictions.
Interpretation: An accuracy of 1.0 means perfect prediction, while 0.5 would be equivalent to random guessing for a balanced dataset.


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

## Loss Functions

The Faster R-CNN model typically uses a multi-task loss function that combines several components:

## 1. Classification Loss

- Type: Cross-Entropy Loss
- Purpose: Measures the error in classifying the object (e.g., car or truck)
- Applied to: Both the Region Proposal Network (RPN) and the final classifier

## 2. Bounding Box Regression Loss

- Type: Smooth L1 Loss (also known as Huber Loss)
- Purpose: Measures the error in predicting the bounding box coordinates
- Applied to: Both the RPN and the final regressor
- Advantage: Less sensitive to outliers compared to standard L2 loss

## 3. Objectness Loss

- Type: Binary Cross-Entropy Loss
- Purpose: Measures the error in predicting whether a region contains an object or not
- Applied to: RPN

## Combined Loss Function

The total loss is a weighted sum of these components:
L_total = λ_cls * L_cls + λ_box * L_box + λ_rpn_cls * L_rpn_cls + λ_rpn_box * L_rpn_box

Where:
- L_cls: Classification loss
- L_box: Bounding box regression loss
- L_rpn_cls: RPN classification loss (objectness)
- L_rpn_box: RPN bounding box regression loss
- λ: Balancing parameters for each loss component

## Evaluation Metrics

The code uses two primary metrics for evaluation: Accuracy and Validation Loss. These are calculated in the `evaluate_model` function.

## 1. Accuracy

The accuracy metric in this implementation is based on the Intersection over Union (IoU) between predicted and ground truth bounding boxes.

### Calculation Process:
1. For each image in the validation set:
   - Compare each ground truth box with all predicted boxes
   - Calculate IoU for each pair
   - A prediction is considered correct if its IoU with a ground truth box exceeds a threshold (not explicitly defined in the code, typically 0.5)

2. Accuracy is then calculated as:
Accuracy = Number of Correct Predictions / Total Number of Ground Truth Boxes


## Computational Resources

### Mask R-CNN

- **Hardware**: Laptop
- **Processing Unit**: Single CPU
- **Training Duration**: 64 hours

The Mask R-CNN model, known for its effectiveness in instance segmentation tasks, was trained on a standard laptop configuration. This setup demonstrates the model's ability to be trained on consumer-grade hardware, albeit with a significant time investment.

### Lane Net

- **Hardware**: Personal Computer (PC)
- **Processing Units**: 
  - GPU (1)
  - CPU (1)

Our custom Lane Net model leveraged both CPU and GPU processing power on a personal computer. This configuration allows for faster training times compared to CPU-only setups, making it suitable for iterative development and experimentation.

### Faster R-CNN

- **Platform**: Google Cloud
- **Processing Units**:
  - GPUs (2)
  - CPU (1)

The Faster R-CNN model was trained using cloud computing resources, specifically on Google Cloud. This high-performance setup with multiple GPUs is ideal for training complex models or working with large datasets, significantly reducing training time compared to local machine setups.

## Training Models

[Provide details on how to train each model, including hyperparameters and training scripts]

## Transfer Learning

### Introduction To Transfer Learning

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

### Application of Transfer Learning in This Project

The aim of our experiment was to check with a line detection model and an object detection model on a simple camera, we were able to predict on the road the direction that a self-driving system should follow, with proximity indicators of other vehicles.

### Transfer Learning Code

To achieve this goal we decided to implement from scratch a LaneNET model and Mask R-CNN model, compare the results and apply the best one.

In addition we tried to implement a Faster R-CNN for object detection, to indicate other vehicles proximity.

## Models Comparison

The first goal was get the lines detection, using two different sizes of image on the Mask R-CNN model devellopped from scratch (180x320 vs 480x854). The model was giving a small output of fixed dimension, causing that recover the original image size was only possible using the 180x320 size input.
We stop the trials due to the long train process (at this point we were training using a laptop with CPU).

The results using a google maps image as reference were as follows:

Mask R-CNN
![alt text](<report images/MaskRCNN.png>)

At the same type we were preparing the second model LaneNET to get a different performance. In this case we train the model focus on images of 480x854, expecting get a better precision. The model was get from a paper but using a different dataset (smaller in fact).

![alt text](<report images/LaneNET v0.png>)

The results were good, but not as good as we expected. At this point we decided adapt the bdd100k Dataset to the model to train it with a lot more samples. The result was surprising, a lot more of precision, just detecting the road lines.

![alt text](<report images/LaneNET v1.png>)

## Validation With Our Own Images

The next step was implement both models in a real video to validate real-world performance. It was done using a mobile in a car, driving on the highway.

<video controls src="report images/MaskRCNN vs LaneNET.mp4" title="Title"></video>

The LaneNet model (right side) was a lot better than the Mask R-CNN (left side), like we expected. the surprising was that both models were able to detect the lines in really bad quality video.

To achieve our goal we had some troubles like:
  - The images size applied to model had to match with the trained images side to get good results.
  - The images focus had to be consistent with train dataset (landscape vs road).
  - Difficult define the threshold to decide if it is point or not, it depends a lot of the image quality and resolution.

Once developed the line detection models, we create a simple algorithm to calculate the angle deviation from previous to current frame on a video, and we applied an addition calculation to rotate a possible wheeldrive. We tried to simulate a possible self-driving system.

This algorithm was based in the following steps:
  - Find the center of the binary mask image.
  - Calculate the image rotation from this point respect the previous image.
  - Apply a simple calculation: angle_new = angle_old + (2 x angle_diff) 

<video controls src="report images/Self Driving.mp4" title="Title"></video>

To give consistence to experiment, we applied also the algorithm to the original image (left side). The results were as we expected, better on the LaneNET predictor (right side), than in the Mask R-CNN (center). The precision of the image, helped a lot to get a good response. 

The last experiment was add a object detection to our system. We implent a Faster R-CNN from ResNet50 and what had to be simple caused as many troubles. We used a annotations from the same dataset bdd100k.

The training of this model required a lot of performance, so required to use a Google Cloud with GPU. We train for first time using 20000 images with 30 epochs, just looking for cars. The results were not like expected, were bad and not repetitive.

![alt text](<report images/FasterRCNN v0.png>)

The train and validate loss didn’t go according, accuracy really bad and overfitting.

![alt text](<report images/FasterRCNN Train v0.png>)

We increase the number of samples using less epochs and fixed the seed to be more deterministic. Also we include in out annotations the trucks.

The next train was also done using a GPU on Google Cloud, with 70000 samples and only 4 epochs, but the results still bad. 

![alt text](<report images/FasterRCNN v1.png>)

Train and validation loss graphs didn't go according and accuraccy was really bad.

![alt text](<report images/FasterRCNN Train v1.png>)

Found that annotations were not properly scaled according to the image resize transform, so we tried to fix it and train it again, bu we expend the rest of Google Cloud balance we had.
We couldn’t finish the tranning (we tried using local CPU but was too long), so we decided implement the pre-trained model.

The results were satisfactory.

![alt text](<report images/FasterRCNN v2.png>)

The final experiment was implement in the self-driving system the object detection and show the proximity of a car by colors:

  - green, the vehicle was far
  - yellow, the vehicle was close
  - red, the vehicle was too close

To decide the color, we use the lower height of the box predicted. The results are visible in the following video.

<video controls src="report images/ObjectDetection & WheelDrive v1.mp4" title="Title"></video>

The experiment was a success.

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
