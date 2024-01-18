import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader
from torchvision import transforms
import torch
from model import predict_model,Dataset
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50
from util import text_preprocess,getfrom_directory,get_texts_from_textsPath
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(), 
])
max_length = 131  # 输入最大长度
num_classes=3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = "data/data/"
batch_size=64
test_path = "test_without_label.txt"
test_df = pd.read_csv(test_path,sep=",")
test_df.iloc[:,-1]=0
test_labels = np.array(test_df['tag'])
image_paths_test = getfrom_directory(folder_path,test_df)
test_texts = get_texts_from_textsPath(folder_path,test_df)
tokenized_texts_test = text_preprocess(test_texts)
dataset_test = Dataset(image_paths_test, tokenized_texts_test, test_labels, transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
best_model = torch.load('multi_model.pt').to(device)
test_predictions = predict_model(best_model, loader_test, device)  
test_predictions = np.array(test_predictions)
column_dict_ = {0:"positive", 1:"negative",2:"neutral"}
test_df['tag'] = test_predictions
pre_df = test_df.replace({"tag": column_dict_})
pre_df.to_csv('predict.txt',sep=',',index=False)
print("prediction finished")