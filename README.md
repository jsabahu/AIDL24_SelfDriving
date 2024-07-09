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
     * [Mean Avergae Precision](#mean-average-precision-map)
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

Students: Jordi Sabatés, Marc Ramon, Marc GIné and Fermin Gomila

Advisor: Daniel Fojo

Please guys create your own git branches (name the branches with your names): git checkout -b [name_of_the_branch]


To check the branch I am currently modifiying: git branch

To check the base branch: git checkout dev

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
