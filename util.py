import cv2
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from torchvision.models import resnet50
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
pretrained_model = BertModel.from_pretrained("bert-base-multilingual-cased")

max_length = 131 
num_classes=3
def getfrom_directory(folder_path ,df):
    image_paths = []
    for ind in df['guid']:
        image_path = folder_path+str(ind)+".jpg"
        try:
            image = cv2.imread(image_path)
            image_paths.append(image_path)
        except Exception as e:
            pass
            continue
    
    return image_paths


def get_texts_from_textsPath(folder_path,df):
    texts=[]
    for ind in df['guid']:
        file = folder_path+str(ind)+".txt"
        try:
            with open(file, "r",encoding="GB18030") as infile:
                content = infile.read()
                texts.append(content)
        except FileNotFoundError:
            continue
    return texts


def text_preprocess(texts):
    tokenized_texts = [tokenizer(text,padding='max_length',max_length=max_length,truncation=True,return_tensors="pt") for text in texts]
    return tokenized_texts