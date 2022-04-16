import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples




def label_smoothing(outputs,epsion = 0.1):
    k  = outputs.shape[1]
    smoothed_label = (1-epsion)*outputs + epsion / k
    return smoothed_label

def confidence_update(model, confidence, batchX, batchY, batch_index):

    with torch.no_grad():
        batch_outputs = model(batchX)
        cav = (batch_outputs*torch.abs(1-batch_outputs))*batchY
        cav_pred = torch.max(cav,dim=1)[1]
        gt_label = F.one_hot(cav_pred,batchY.shape[1]) # label_smoothing() could be used to further improve the performance for some datasets
        confidence[batch_index,:] = gt_label.float()

    return confidence


