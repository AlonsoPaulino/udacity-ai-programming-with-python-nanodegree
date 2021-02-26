# PROGRAMMER: Luis Alonso Paulino Flores
# DATE CREATED: Feb, 25, 2021.
# REVISED DATE: -
# BASIC USAGE: python predict.py flowers/test/1/image_06743.jpg --gpu

import argparse 
import time
import sys
import json
import numpy as np
import torch

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
    
    parser.add_argument('image_path', type=str, help='image path from the image you want to evaluate (mandatory)')
    parser.add_argument('--checkpoint', type=str, default='check.pth', help='model checkpoint previously stored in train.py (.pth)')
    parser.add_argument('--top_k', type=int, default=5, help='number of top results to show (5 by default)')
    parser.add_argument('--cat_file', type=str, default='cat_to_name.json', help='file that maps labels with its category name (.json)')
    parser.add_argument('--gpu', action='store_true', help='enables gpu')
    
    in_args = parser.parse_args()

    if in_args.image_path is None:
        raise Exception("you should enter the image_path from the image you want to evaluate")
    if in_args.gpu and not torch.cuda.is_available():
        raise Exception("--gpu not available")

    print(str(in_args))

    return in_args


# Retrieve model and the idx_to_class dict previously stored along with the checkpoint 
# idx_to_class will be useful for displaying the right category name later)
def get_model_and_idx_to_class():
    checkpoint = torch.load(in_args.checkpoint)
    
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    idx_to_class = checkpoint['idx_to_class']
    
    return model, idx_to_class


# Process the image and convert it into a numpy array
def transform_to_np_array(image):
    # Using Pillow to open the requested image
    img = Image.open(image)
    width, height = img.size
    
    coordinates = [width, height]
    max_element = coordinates.index(max(coordinates))

    if (max_element == 0):
        min_element = 1
    else:
        min_element = 0
    
    aspect_ratio = coordinates[max_element] / coordinates[min_element]
    
    # Calculate image new coordinates by keeping the original aspect ratio
    new_coordinates = [0,0]
    new_coordinates[min_element] = 256
    new_coordinates[max_element] = int(256 * aspect_ratio)
    
    # Resize the image
    img = img.resize(new_coordinates)   
    width, height = new_coordinates
    
    # Calculate crop box making sure the same area is cut for left-right and top-bottom
    left = (width - 244) / 2
    right = (width + 244) / 2
    top = (height - 244) / 2
    bottom = (height + 244) / 2
    
    # Apply crop to get a 244 x 244 square image
    img = img.crop((left, top, right, bottom))

    # Create np_array_from_image
    np_array_from_image = np.array(img)
    np_array_from_image = np_array_from_image.astype('float64')
    np_array_from_image = np_array_from_image / [255,255,255]
    np_array_from_image = (np_array_from_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_array_from_image = np_array_from_image.transpose((2, 0, 1))
    
    return np_array_from_image


# Get classification results
def get_results():
    image_path = in_args.image_path
    top_k = in_args.top_k
    cat_file = json.loads(open(in_args.cat_file).read())

    with torch.no_grad():
        # Load image
        image = transform_to_np_array(image_path)
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()

        # Load model and idx_to_class to get the labels
        model, idx_to_class = get_model_and_idx_to_class()

        if (in_args.gpu):
            image, model = image.cuda(), model.cuda()
        else:
            image, model = image.cpu(), model.cpu()
        
        # Use PyTorch to get the top k classification results
        top_probs, top_classes = torch.exp(model(image)).topk(top_k)
        
        # Match classes with their own category label from cat_file
        top_labels=[]
        
        for c in top_classes.cpu().numpy().tolist()[0]:
            top_labels.append(idx_to_class[c])

        top_flowers = [cat_file[str(lab)] for lab in top_labels]
    
    return zip(top_probs[0].cpu().numpy(), top_labels, top_flowers)
        

# Show classification results
def show_results(results):
    i = 0
    for prob, _, flower in results:
        i = i + 1
        formatted_prob = "{} %".format(str(round(prob, 4) * 100.))
        print("{}.{} ({})".format(i, flower, formatted_prob))

       
# Perform the prediction with all the input arguments received in the console
in_args = get_input_args()
results = get_results()
show_results(results)