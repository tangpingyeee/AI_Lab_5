import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader
from model import Dataset,FusionModel,predict_model,train
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50
from util import text_preprocess,getfrom_directory,get_texts_from_textsPath
max_length = 131  # 输入最大长度
num_classes=3
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(), 
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#其实应该用gpu，但是我的电脑配置不太行，用cpu的结果是高精度情况下跑了整整十个小时（我真是哔了狗了）
folder_path = "data/data/"
train_label_path = "train.txt"
train_label_df = pd.read_csv(train_label_path,sep=",")
column_dict = {"positive": 0, "negative": 1,"neutral":2}
new_df = train_label_df.replace({"tag": column_dict})
labels = list(new_df['tag'])

image_paths = getfrom_directory(folder_path,new_df)
texts = get_texts_from_textsPath(folder_path,new_df)

# 验证集
image_paths_train, image_paths_val, texts_train, texts_val, labels_train, labels_val = train_test_split(
    image_paths, texts, labels, test_size=0.2, random_state=5)
#文本预处理
tokenized_texts_train = text_preprocess(texts_train)
tokenized_texts_val = text_preprocess(texts_val)
# 数据集
dataset_train = Dataset(image_paths_train, tokenized_texts_train, labels_train, transform)
dataset_val = Dataset(image_paths_val,tokenized_texts_val, labels_val, transform)


batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
lrlist = [1e-5,3e-5]
batch_size = 64
best_acc = 0
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

print("start training")
for lr in lrlist:
    model = FusionModel(num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    num_epochs = 6
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, loader_train, criterion, optimizer, device)
        val_predictions = predict_model(model, loader_val, device)   
        val_predictions = np.array(val_predictions)
        val_labels = np.array(labels_val)
        val_acc = (val_predictions == val_labels).sum() / len(val_labels)
        if(val_acc>best_acc):#保存比较好的模型
            best_acc = val_acc
            torch.save(model, 'multi_model.pt')
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Best Val Acc:{best_acc:.4f}")
print("training finished")