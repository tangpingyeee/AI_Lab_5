import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import argparse
from torchvision.models import resnet50
from util import text_preprocess,getfrom_directory,get_texts_from_textsPath
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")
class ImageFeatureExtractor(nn.Module):
    def __init__(self):
        super(ImageFeatureExtractor, self).__init__()
        self.resnet = resnet50(pretrained=True)  
    
    def forward(self, image):
        features = self.resnet(image)
        return features

class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert = pretrained_model

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]  
        output = pooled_output
        return output
    

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.image_extractor = ImageFeatureExtractor()  
        self.text_encoder = TextFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1768, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
            nn.ReLU(inplace=True),
        )
    def forward(self, image, input_ids,attention_mask):
            image_features = self.image_extractor(image)
            text_features = self.text_encoder(input_ids,attention_mask)
            fusion_features = torch.cat((text_features,image_features), dim=-1)
            output = self.classifier(fusion_features)
            return output
def train(model,loader,criterion, optimizer, device):
    model.train()
    loss=0
    correct=0
    for images, input_ids, attention_mask, labels in loader:
        optimizer.zero_grad() 
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)     
        labels = labels.to(device) 
        outputs = model(images, input_ids,attention_mask)
        preds = torch.max(outputs, 1)
        correct =correct+ torch.sum(preds[1] == labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()   
        loss = loss+loss.item()
    epoch_loss = loss / len(loader)
    epoch_acc = correct.item() / len(loader.dataset)
    return epoch_loss, epoch_acc
def predict_model(model, test_loader, device):
    model.eval()
    predictions = []
    for images,input_ids, attention_mask,  _ in test_loader:
        images = images.to(device)
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.to(device)
        with torch.no_grad():
            outputs = model(images, input_ids,attention_mask)
            _, preds = torch.max(outputs, 1)
        predictions.extend(preds.cpu().numpy())
    return predictions

    