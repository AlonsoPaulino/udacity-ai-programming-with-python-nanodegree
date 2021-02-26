# PROGRAMMER: Luis Alonso Paulino Flores
# DATE CREATED: Feb, 25, 2021.
# REVISED DATE: -
# BASIC USAGE: python train.py flowers --epochs=1 --arch=vgg13 --gpu

import argparse
import os
import time
import json
import workspace_utils
import torch
import matplotlib.pyplot as plt 
import numpy as np

from collections import OrderedDict
from workspace_utils import active_session
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

'''
    # Resources
    References used as guidance for this implementation: 
    - https://medium.com/@josh_2774/deep-learning-with-pytorch-9574e74d17ad
''' 


# Read all arguments from the console
def get_input_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_path', type=str, default='flowers', help='data directory path (mandatory)')
    parser.add_argument('--checkpoint', type=str, default='check.pth', help='checkpoint file to save the model (.pth)')
    parser.add_argument('--arch', type=str, default='vgg', help='arch used to build the model (by default vgg) OPTIONS[vgg13, vgg16, vgg19, densenet]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate (0.001 by default)')
    parser.add_argument('--hidden_units', type=int, default=4096, help='# of hidden units in the model (4096 by default)')
    parser.add_argument('--epochs', type=int, default=10, help='# of iterations required for training (10 by default)')
    parser.add_argument('--gpu', action='store_true', help='enables gpu')
    
    in_args = parser.parse_args()

    if (in_args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu not available")
    if in_args.arch not in ('vgg13','vgg16', 'vgg19', 'densenet', None):
        raise Exception('arch is not supported')  

    print(str(in_args))

    return in_args


# Read data that comes from data_patch and apply transformations according to
# what is stated in the projects requirements
def get_flowers_data():
    crop_size = 224
    # Constants used for means normalization
    norm_means = (0.485, 0.456, 0.406)
    # Constants used for standard deviations normalization
    norm_sd = (0.229, 0.224, 0.225)

    # Define transformations according to the project requirements
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_means, norm_sd),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_means, norm_sd),
    ])

    train_data_path = in_args.data_path + '/train'
    test_data_path = in_args.data_path + '/test'
    valid_data_path = in_args.data_path + '/valid'

    # Apply transformations
    train_dataset = datasets.ImageFolder(train_data_path, transform=train_transforms)
    test_dataset  = datasets.ImageFolder(test_data_path, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_data_path, transform=test_transforms)

    # Save each data set in a dictionary upon creation
    dict_image_datasets = {
        'train': train_dataset,
        'test': test_dataset,
        'valid': valid_dataset
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    # As suggested during code review, suffling isn't required for either test dataset nor valid dataset
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Save each data loader in a dictionary upon creation
    dict_loaders = {
        'train': train_loader,
        'test': test_loader,
        'valid': valid_loader
    }

    return dict_loaders, dict_image_datasets


# Create model using the arch specified (vgg16/vgg13/vgg19/densenet)
def create_model():
    if (in_args.arch == 'vgg16'):
        model = models.vgg16(pretrained=True)
        input_node=25088
    elif (in_args.arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
        input_node=25088
    elif (in_args.arch == 'vgg19'):
        model = models.vgg19(pretrained=True)
        input_node=25088
    elif (in_args.arch == 'densenet'):
        model = models.densenet(pretrained=True)
        input_node=1024
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_node, in_args.hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(in_args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier

    return model


# Return device spec based on whether GPU is enabled or not
# If GPU is enabled, we proceed to use cuda, otherwise we just rely on the cpu
def get_device():
    if (in_args.gpu):
        device = 'cuda'
    else:
        device = 'cpu'

    return device


# Train model using the dict_loaders which holds each data set type (train, test and valid)
def train_model(model, dict_loaders, print_freq=10):
    train_loader = dict_loaders['train']
    test_loader = dict_loaders['test']
    valid_loader = dict_loaders['valid']
    
    device = get_device()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)
    
    step = 1
    model.to(device)
    
    print("Training model with arch {} has started".format(in_args.arch))

    for e in range(in_args.epochs) :
        running_loss = 0
        for _, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if step == 1 or step % print_freq == 0:
                valid_accuracy = get_model_accuracy(model, valid_loader)
                print("Epoch: {}/{}   |   ".format(e + 1, in_args.epochs),
                      "Step: {}   |    ".format(step),
                      "Loss: {:.4f}   |   ".format(running_loss / print_freq),
                      "Validation Accuracy: {}".format(round(valid_accuracy,4)))
                running_loss = 0

            step += 1

    print("Training is complete")
    
    test_result = get_model_accuracy(model, test_loader)
    
    print('Model with arch {} ends up with accuracy: {}'.format(in_args.arch, test_result))

    return model


# Get model accuracy (calculated by #images evaluated correcty / #total of images).
def get_model_accuracy(model, test_loader):
    correct = 0
    total = 0
    device = get_device()

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total  


# Create and save the checkpoint corresponding to the already created and trained model
# Additionaly, save some useful information that is useful such as the idt_to_class mapping 
# using the train dataset
def create_checkpoint(model, dict_image_sets):
    train_image_set = dict_image_sets['train']
    dict_idx_to_class = { idx: key for key, idx in train_image_set.class_to_idx.items() }

    checkpoint = {
        'model': model.cpu(),
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'idx_to_class': dict_idx_to_class,
    }

    torch.save(checkpoint, in_args.checkpoint)
    
    
# Perform the model creation and training with all the input arguments received in the console
# Using active session utils to prevent workspace to get idle and session to expire
with active_session():
    in_args = get_input_args()
    dict_loaders, dict_image_sets = get_flowers_data()
    
    model = create_model()
    model = train_model(model, dict_loaders)

    create_checkpoint(model, dict_image_sets)

