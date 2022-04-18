# CAVL (ICLR2022) 
This is an official implementation of [Exploiting Class Activation Value for Partial-Label Learning](https://openreview.net/forum?id=qqdXHUGec9h), which is accepted by ICLR2022 poster.

## Abstract 
Partial-label learning (PLL) solves the multi-class classification problem, where each training instance is assigned a set of candidate labels that include the true label. Recent advances showed that PLL can be compatible with deep neural networks, which achieved state-of-the-art performance. However, most of the existing deep PLL methods focus on designing proper training objectives under various assumptions on the collected data, which may limit their performance when the collected data cannot satisfy the adopted assumptions. In this paper, we propose to exploit the learned intrinsic representation of the model to identify the true label in the training process, which does not rely on any assumptions on the collected data. We make two key contributions. As the first contribution, we empirically show that the class activation map (CAM), a simple technique for discriminating the learning patterns of each class in images, could surprisingly be utilized to make accurate predictions on selecting the true label from candidate labels. Unfortunately, as CAM is confined to image inputs with convolutional neural networks, we are yet unable to directly leverage CAM to address the PLL problem with general inputs and models. Thus, as the second contribution, we propose the class activation value (CAV), which owns similar properties of CAM, while CAV is versatile in various types of inputs and models. Building upon CAV, we propose a novel method named CAV Learning (CAVL) that selects the true label by the class with the maximum CAV for model training. Extensive experiments on various datasets demonstrate that our proposed CAVL method achieves state-of-the-art performance.

## Prerequisite
* The requirements are in **requirements.txt**. However, the settings are not limited to it (CUDA 11.0, Pytorch 1.7 for one RTX3090). 

## Usage
1. Train the model by running the following command directly. Remember to set the chosen dataset, model backbone and hyper-parameters in the script. 
    ```
    python main.py
    ```
