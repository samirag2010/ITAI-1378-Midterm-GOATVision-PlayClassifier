G.O.A.T Vision LIVE DEMO 
https://goatvision-soccer.streamlit.app

## 📸 Demo

![G.O.A.T Vision Demo](docs/demo.png)

ITAI-1378 Midterm – G.O.A.T Vision: Game Outcome Action Tracker
Project Name

G.O.A.T Vision (Game Outcome Action Tracker) is a computer vision application that classifies soccer plays from images using a deep learning model (ResNet50). Users can upload an image and receive a prediction along with confidence scores.

Course: ITAI 1378 – Computer Vision
Student: Samira Gevara

Project Tier

Tier 2 – Custom Dataset + Transfer Learning

This project builds a custom image classification system using transfer learning with a pretrained deep learning model. 

Problem Statement

Analyzing soccer events from visual footage is often a manual and time-consuming process for analysts, coaches, and content creators. Automatically identifying key moments such as free kicks, penalty kicks, corner kicks, shots, and yellow cards can significantly improve the efficiency of highlight creation and match analysis.

This project explores how computer vision can classify soccer event images into specific play types.

Solution Overview

G.O.A.T Vision is a computer vision model that classifies soccer event images into five categories:

Corner Kick

Free Kick

Penalty Kick

Shot on Goal

Yellow Card

Displays prediction confidence

Visual probability breakdown

The system uses transfer learning with a pretrained ResNet50 model implemented in PyTorch. Images are preprocessed and passed through the neural network to produce a predicted soccer event class.

Technical Approach

Technique: Image Classification

Model: ResNet50 (pretrained on ImageNet)

Frameworks and Tools:

PyTorch

torchvision

scikit-learn

Google Drive (model hosting via gdown)

Streamlit

matplotlib

Google Colab (GPU)

Transfer learning allows the model to leverage features learned from millions of images while adapting to a smaller soccer-specific dataset.

Dataset Plan

Dataset Size: 146 curated images

Classes:

corner_kick

free_kick

penalty_kick

shot

yellow_card

Images were collected from publicly available soccer imagery sources and manually cleaned to remove irrelevant or ambiguous images such as cartoons, logos, or unclear events.

Deployment Notes

Due to GitHub file size limits, the trained model is hosted externally on Google Drive.
The app automatically downloads the model at runtime using gdown.

Data Augmentation Used:

Horizontal flip

Color jitter

Image resizing

Image normalization

These augmentations help improve generalization despite the small dataset.

Model Evaluation

Primary Metric: Validation Accuracy

Target Accuracy: ≥ 70%
Achieved Accuracy: 82.14%

Secondary Evaluation:

Confusion Matrix analysis

Prediction visualization on validation images

Key observations from evaluation:

Yellow card detection achieved perfect classification in validation samples.

Some confusion occurred between shots, penalty kicks, and corner kicks due to similar goal-area visual contexts.

System Pipeline
Input Soccer Image
        ↓
Image Preprocessing
(resize, normalization, augmentation)
        ↓
ResNet50 Feature Extraction
(transfer learning)
        ↓
Fully Connected Classifier
        ↓
Predicted Event Label
(Corner Kick | Free Kick | Penalty Kick | Shot | Yellow Card)

Week-by-Week Plan
Week	Task	Milestone
Week 11	Dataset collection and cleaning	Dataset prepared
Week 12	Model setup and preprocessing	Training pipeline ready
Week 13	Train model and test predictions	Model learning
Week 14	Evaluate results and confusion matrix	Performance analysis
Week 15	Prepare slides and documentation	Presentation ready
Week 16	Present project	Final submission (May 8th)

Resources Needed
Resource	Details
Compute	Google Colab GPU
Frameworks	PyTorch, torchvision
Libraries	sklearn, matplotlib
Estimated Cost	$0

Risks and Mitigation
Risk	Probability	Mitigation
Small dataset size	Medium	Data augmentation and transfer learning
Ambiguous images	Medium	Manual dataset cleaning
Class confusion	Low	Model fine-tuning and confusion matrix analysis

AI Usage Log

AI tools such as ChatGPT were used to assist with:

project planning

code debugging

PyTorch implementation guidance

documentation drafting

All code execution, dataset preparation, and model experimentation were performed by the student (Samira Gevara)

Repository Structure
project-name/
README.md
requirements.txt
notebooks/
    01_exploration.ipynb
data/
    README.md
docs/
    proposal.pdf

Future Improvements

Potential improvements for future development include:

expanding the dataset to thousands of labeled soccer event images

training with video frame sequences instead of single images

implementing object detection to localize players and referees

building a real-time match event detection system







