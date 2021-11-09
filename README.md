# Cakes and Tensors Facial Recognition Project

Mentee: Seth Billiau

Mentor: Ryan Caulfield

## Goals

This application performs facial recognition on photos in my photo library with the goal of recognizing the faces of my friends and family. Though this certainly isn't a novel problem (Google photos already does this very well), I hope to build important skills through the process of building this app. I also hope to use this as an opportunity to explore some topics in computer vision that are of interest to me and are hot topics in the literature. 

Learning goals:  
- I know how to develop containerized applications using Docker.
- I know how to use common computer vision libraries like opencv-python and how to adapt open source tools and solutions like [MTCNN](https://arxiv.org/abs/1604.02878%20%22https://arxiv.org/abs/1604.02878) and [FaceNet](https://arxiv.org/abs/1503.03832) to solve problems. 
- Build ML pipelines for training and inference.
- I know how to use Edge hardware on my own [Jetson Nano](https://developer.nvidia.com/embedded/jetson-nano) and by ssh'ing into hardware owned by the Booz Allen Edge AI Lab for computer vision. 

Extensions: 
- I explore how Transformer models like [Hugging Face's ViT](https://huggingface.co/transformers/model_doc/vit.html) can be used for object detection and image classification, and how these models can be specifically adapted for facial recognition and face detection (month 3). 
- I explore how model interpretation and explainability tools like LIME and Integrated Gradients can be used to demystify facial recognition systems (month 3). 

## Timeline

Week 1
- 9/13: Ryan Caulfield and Seth Billiau have an introductory meeting to discuss the C&T mentorship program and discuss computer vision as a common interest. 
- 9/16: Seth attends the C&T Mentee training with the C&T Mentorship cohort led by Wesley Clark. Wesley outlines program expectations and adds us to a slack channel where we can share ideas. 
- 9/17: Ryan and Seth meet to discuss 3 potential project ideas that Seth brainstormed during the week. Seth decides that he wants to develop software development skills, so we decide to narrow our focus to developing a small, self-contained product. We leave the meeting with a commitment to develop a more mature product proposal in week 2. 

Week 2
- 9/23: Ryan and Seth develop a weekly meeting cadence on Thursdays where Seth will present code that he has worked on, Ryan can address potential roadblocks, and both of them can establish next steps for the week. During this week's meeting, Seth presents developed proposals for 3 projects: Medical Imaging with Transformer Models, Facial Recognition with an Edge device, and auto vehicle detection and classification with an Android Studio app. After discussing each of the 3 options, we decide to develop the facial recognition project. Ryan reaches out to Mike Sellers and JT Wolohan at the Edge AI Lab to explore the possibility of a collaboration with that location in Belcamp, Maryland. 
- 9/24: Ryan, Seth, and Mike Sellers meet to discuss the potential for Seth to use the Edge AI lab for the project. Mike is excited about developing a new user persona for the lab and anticipates that Seth can ssh into devices in the lab for inference once the product is developed.

Week 3
-  9/30: Ryan and Seth use this week's stand up as an opportunity for Ryan to give Seth a crash course in Docker and git etiquette. Ryan walks Seth through a Dockerfile that he has used in the past for a computer project on a USPS contract. Seth uses this Dockerfile as a template to create a baseline container image for the product. Ryan also reviews lots of basic git etiquette with Seth to help him develop as a software engineer. Seth creates a GitHub repository for his project, and uses the project as an opportunity for him to practice commiting code to branches, reviewing pull requests, and collaborating on a code base with others. 
- 10/1: Seth and Ryan begin to collect training data for the project (pictures of themselves and others). Seth begins to write code to develop a pipeline for facial recognition to parse images, find faces, and train a classification model to assign labels to each face. 

Week 4
- 10/7: Seth has written scripts to convert images from .HEIC format to .jpg format and to ingest images as numpy arrays. He has begun working on face detection, but hasn't gotten very far. Seth asks Ryan to do some research on face detection libraries, identifying classification as the primary area of focus for the project. 

Week 5
- 10/14: Seth has finished implementing face detection using the MTCNN library that uses CNNs to identify faces in training images. However, the faces are not always identified correctly. Seth and Ryan begin brainstorming ways to eliminate faces that are not commonly found in the training data or false positives identified by the MTCNN library (images that were incorrectly identified as faces). Seth and Ryan settle on an idea: use an autoencoder to reconstruct images of faces, then chop off the decoder so that you have a sparse latent space representation of the inputs. Then, use this sparse latent space to perform clustering to recognize commonly seen faces or perform anomaly detection to identify false positives.
- Weekend of 10/15: Seth uses google colab to train linear and convolutional autoencoders for faces on the [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/). Unfortunately after experimenting with [sklearn's implementations of KMeans and DBSCAN](https://scikit-learn.org/stable/modules/clustering.html), there is no evidence that clustering in the latent space can effectively identify commonly-observed people. [Random Cut Forest](https://github.com/kLabUM/rrcf)/[Isolation forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) for anomaly detection show some promise in identifying false positives, but for now, we decide to table this exploration and pivot to labeling the input data using [pigeon](https://github.com/agermanidis/pigeon). 

Week 6: 
- 10/21: Seth and Ryan identify a state-of-the-art [dlib](http://dlib.net/) facial recognition library based on FaceNet to create sparse, 128-dimension encodings for faces, along with support for face detection. We opt to continue using MTCNN for our own face detection, but use this library's face encodings to perform facial recognition. Seth completes the training pipeline by creating k-nn classifier in the 128-dimensional latent space. This should be fast enough to perform live motion video inference. Seth begins to create an inference pipeline for facial recognition with his MacBook Pro web camera. 
