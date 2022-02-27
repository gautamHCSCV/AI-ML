import os
import pandas as pd
import numpy as np
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision
import sklearn.metrics as metrics
from PIL import Image
import torch.nn.functional as F


criterion = nn.CrossEntropyLoss()


# In[3]:


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


# In[4]:


path0 = '../../../dataset/Stanford_chestxray_dataset/'
path = '../../../dataset/Stanford_chestxray_dataset/CheXpert-v1.0-small/'
os.listdir(path)


# In[5]:


path1 = path + 'train.csv'
df = pd.read_csv(path1)
df.head()


# In[11]:


mapping = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Pleural Effusion': 3, 'No Finding': 4, 'Pneumothorax': 5}
disease_labels = {j:i for i,j in mapping.items()}
diseases = ['Consolidation','Cardiomegaly','No Finding','Pleural Effusion','Pneumothorax','Atelectasis']
print(disease_labels)


# In[7]:


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# In[10]:


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# In[25]:


def Evaluate(model):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0
    preds = []
    pred_labels = []
    labels = []
    with torch.no_grad():
        for i in range(len(df)):
            if df['Frontal/Lateral'][i]=="Lateral":
                continue
            for j in diseases:
                if df[j][i]==1.0:
                    image = pil_loader(path0+df['Path'][i])
                    x = data_transforms['test'](image)
                    x = torch.Tensor(np.expand_dims(x,axis = 0))
                    x = x.to(device)
                    valid_logits = model(x)
                    predict_prob = F.softmax(valid_logits)

                    _,predictions = predict_prob.max(1)
                    predictions = predictions.to('cpu')
                    prediction = int(predictions[0])
                    if df[disease_labels[prediction]][i] and df[disease_labels[prediction]][i]== 1:
                        running_corrects += 1
                        labels.append(prediction)
                    else:
                        labels.append(mapping[j])
                    predict_prob = predict_prob.to('cpu')

                    pred_labels.extend(list(predictions.numpy()))
                    preds.extend(list(predict_prob.numpy()))
                    total += 1
                    break
        print('Accuracy:',running_corrects/total)
        return(np.array(preds), np.array(pred_labels),np.array(labels))
    
    
def ROC_plot(y_probas,labels,name = 'abc.svg'):
    for c in range(6):
        fpr = []
        tpr = []
        thresholds = np.arange(0.0, 1.01, .01)

        P = list(labels).count(c)
        N = len(labels) - P

        for thresh in thresholds:
            FP=0
            TP=0
            for i in range(len(labels)):
                if (y_probas[i][c] > thresh):
                    if labels[i] == c:
                        TP = TP + 1
                    else:
                        FP = FP + 1
            fpr.append(FP/float(N))
            tpr.append(TP/float(P))
            
        auc = np.trapz(tpr,fpr)
        print('\tclass',c,'auc',auc)
        plt.plot(fpr, tpr, label = 'Class: {}, auc:{:.3f}'.format(c,auc))
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE')
    plt.savefig(name, format = 'svg')
    plt.show()
    plt.clf()



print('\nMobilenet-V2')

torchmodel = torchvision.models.mobilenet_v2(pretrained = True)
torchmodel.classifier[1] = nn.Linear(in_features=1280, out_features=6, bias=True)
model = torchmodel
pa = '../../../dataset/VINBIG_DATA/stanford_files/'

model.load_state_dict(torch.load(pa + 'mobilenetv2_6.pth', map_location=device)['model'])
model = model.to(device)

preds, pred_labels,labels = Evaluate(model)
print(metrics.precision_recall_fscore_support(np.array(labels), np.array(pred_labels)))
ROC_plot(preds,labels,'mobilenet_v2.svg')
print(metrics.roc_auc_score(np.array(labels), np.array(preds), multi_class='ovr'))
print(metrics.classification_report(labels,pred_labels))


    
print('\nShufflenet')
torchmodel = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
torchmodel.fc = nn.Linear(in_features=1024, out_features=6, bias=True)
model = torchmodel
model.load_state_dict(torch.load(pa + 'shuffle5v1.pth', map_location=device)['model'])
model = model.to(device)

preds, pred_labels,labels = Evaluate(model)
print(metrics.precision_recall_fscore_support(np.array(labels), np.array(pred_labels)))
ROC_plot(preds,labels,'shufflenet.svg')
print(metrics.roc_auc_score(np.array(labels), np.array(preds), multi_class='ovr'))
print(metrics.classification_report(labels,pred_labels))




