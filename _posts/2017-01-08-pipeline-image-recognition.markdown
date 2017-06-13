---
layout: post
title:  "Machine Learning in Visual Recognition"
date:   2017-01-08 11:20:15 +0800
categories: main
---
<h1>Introduce</h1>
Machine Learning can be used in various different areas. In this project I'll show how machine learing can be applied in the image recognition. Specifically, it will be introduced how the electricity meter will be automatically read. This project was implemented with OpenCV. Normally the gas or electricity meter looks like in the following figure.
<div class="fig figcenter fighighlight">
  <img src="{{ site.github.url }}/assets/plate.png" width="50%">
  <div class="figcaption">Electricity plate</div>
</div>

Firstly the image area in yellow box will be detected, after it each single number image will be segmented. At last the number image will be classified. This is the basic concept. Now let's go throught it out setp by step. In this project I'll show the last part of this work as a image recognition task. The first and two parts belong to object detection and will be postponed in following blogs. 

<h1>Pipeline of Visual Recognition</h1>
In the classical computer vision or OCR problem, we usually works as follows:
<h2> Data preprocessing</h2>
The training data (images) will be normlized, i.e. noise reduce, ratio adjustment etc. Ususally the images have different illumination, it can also be processed in this step. It requires the image processing approaches in this process.

<h2> Feature Extraction</h2>
After preprocessing the necessary information should be retained as much as possible, cause this information is very importent for training our models. The goal of feature extraction is to make the variablity of templates in the same classes minimized and maximied in different classes. In this prcoess the necessary knowledge of image processing should be required.

<h2> Model Tranining</h2>
The extracted relevant information will be used for training classification model. The machine learning models have different performance variously. So the model should be fairly evluated under some criteria.



The following sections will present each part of this pipeline and you'll get a picture how this state of art process works separately.
