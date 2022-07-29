#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Importing relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
import os
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
import warnings 
warnings.filterwarnings('ignore')


# In[61]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


# In[62]:

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# Reading the Data
df = pd.read_csv("Data/Anomaly_data/244B_15dec2020_selectd.csv")
df.columns = ['DateTime','RS4B_ROAST.Campaign','RS4B_ROAST.Batch','RS4B_ROAST.Lot','244B_XV114.Out2','244B.CoffeeTargetTemperature',
              '244B_RTOComms.TE8R051','244B_SC220.OutputVoltage','244B_RTOComms.XV285','244B_FCV110.Output','244B_Sequences.Roasting',
              '244B_FCV110.Feedback','244B_VST220.AIn','244B_CO270.AIn','244B_TE118.AIn','244B_TE117.AIn','244B_TE116.AIn',
              '244B_SC220.SpdFB','244B_SC220.SpdSP','244B_SC150.SpdSP','244B_RTOComms.XV2853','244B_FCV127.Output','244B_FCV129.Feedback',
              '244B_RTOComms.XV2854','244B_FCV129.Output','244B_FCV545.Output','244B_SC220.SpdHz','244B_PDT100.AIn','244B_TE202.AIn',
              '244B_FQTI200.AIn','244B_TE200.AIn','244B_MachineModes.ModeMaster','244B_MachineModes.ModeDrum']
df.dropna(axis=0,inplace=True)
df["DateTime"]=pd.to_datetime(df.DateTime)
df.index = df.DateTime
df.drop(['DateTime'],axis=1,inplace = True)
print(df.shape)

# In[63]:


df = df[(df['244B_MachineModes.ModeDrum'] >= 230) & (df['244B_MachineModes.ModeDrum'] <= 250) & (df['244B_MachineModes.ModeMaster'] == 30)] 
df.drop(['244B_MachineModes.ModeMaster','244B_MachineModes.ModeDrum','244B_XV114.Out2'],axis=1,inplace=True)

print(df.shape)


# In[64]:
from sklearn.preprocessing import LabelEncoder
enc1 = LabelEncoder()
df['cycle'] = enc1.fit_transform(df['RS4B_ROAST.Batch'])
enc = LabelEncoder()
df['batch'] = enc.fit_transform(df['RS4B_ROAST.Lot'])
enc2 = LabelEncoder()
df['coffee_type'] = enc2.fit_transform(df['RS4B_ROAST.Campaign'])
df.head()

df = df.resample("0.7S").median()
df.dropna(axis=0,inplace=True)
print(df.shape)


to_scale = []
for i in df.columns:
    if df[i].dtype != 'object' and max(df[i])-min(df[i])>100:
        to_scale.append(i)
print(to_scale)


# In[66]:


scaler = MinMaxScaler(feature_range=(0, 100))
df[list(to_scale)] = scaler.fit_transform(df[list(to_scale)])


# # Auto-Encoder model

# In[11]:


class Autoencoder(nn.Module):
    def __init__(self, output_len = 14,**kwargs):
        super().__init__()
        self.output_len = output_len
        self.batch,self.a,self.b = kwargs["input_shape"]
        self.layer_dim = 3
        self.hidden_dim = 16
        self.encoder = nn.LSTM(self.b, hidden_size = self.hidden_dim, num_layers=self.layer_dim, batch_first=True)
        self.hidden = nn.Linear(self.hidden_dim*self.a, 16*self.a)
        self.decoder = nn.Linear(16*self.a, self.output_len*self.a)

    def forward(self, x):
        batch = x.shape[0]
        x, (hn,cn) = self.encoder(x)
        x = x.reshape((batch,-1))
        #print(x.shape)
        x = self.hidden(x)
        x = self.decoder(x)
        out = x.view(batch,self.a,self.output_len)
        return out



model = Autoencoder(input_shape=(1,975,27))
inp = torch.randn(1,975,27)
out = model(inp)
print(out.shape)


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('num parameters',pytorch_total_params,'\n')


# In[69]:


input_column = ['244B.CoffeeTargetTemperature', '244B_RTOComms.TE8R051',
       '244B_SC220.OutputVoltage', '244B_RTOComms.XV285', '244B_FCV110.Output',
       '244B_Sequences.Roasting', '244B_FCV110.Feedback', '244B_VST220.AIn',
       '244B_CO270.AIn', '244B_TE118.AIn', '244B_TE117.AIn', '244B_TE116.AIn',
       '244B_SC220.SpdFB', '244B_SC220.SpdSP', '244B_SC150.SpdSP',
       '244B_RTOComms.XV2853', '244B_FCV127.Output', '244B_FCV129.Feedback',
       '244B_RTOComms.XV2854', '244B_FCV129.Output', '244B_FCV545.Output',
       '244B_SC220.SpdHz', '244B_PDT100.AIn', '244B_TE202.AIn',
       '244B_FQTI200.AIn', '244B_TE200.AIn', 'batch']

to_reconstruct = ['244B_TE116.AIn', '244B_TE117.AIn', '244B_TE118.AIn','244B_TE200.AIn', '244B_TE202.AIn',
                '244B_CO270.AIn', '244B_PDT100.AIn','244B_FCV127.Output','244B_FCV129.Output', '244B_FCV545.Output',
       '244B_SC220.SpdHz','244B_VST220.AIn', '244B_SC150.SpdSP','244B_FQTI200.AIn']

target_column = []

for i in df.columns:
    if i in to_reconstruct:
        target_column.append(i)
        
print(len(target_column),len(input_column))



# Data preparation
i=st=count = 0
c = df.iloc[i]['cycle']
b = df.iloc[i]['batch']
l = 950

x_data = []
y_data = []

while i<len(df):
    if df.iloc[i]['batch']!=b:
        b = df.iloc[i]['batch']
        x_data.append(np.array(df[input_column][i-l:i]))
        y_data.append(np.array(df[target_column][i-l:i]))
        st = i
        
    if df.iloc[i]['cycle']!=c:
        if i-st>=l: 
            x_data.append(np.array(df[input_column][i-l:i]))
            y_data.append(np.array(df[target_column][i-l:i]))
        c = df.iloc[i]['cycle']
        st = i
    i+=1
x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape,y_data.shape)


# In[ ]:

class Custom_data(Dataset):
    def __init__(self, x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.x[item],self.y[item]
train_data = Custom_data(x_data,y_data)
test_data = Custom_data(x_data[100:160],y_data[100:160])

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


# In[ ]:


a = iter(train_loader)
b = next(a)
print(b[0].shape,b[1].shape)


# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = Autoencoder(output_len=b[1].shape[2],input_shape = b[0].shape)
model = model.to(device)

# create an optimizer object
# Adam optimizer with learning rate 1e-3
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# mean-squared error loss
criterion = nn.MSELoss()


# In[22]:


def train_model(model,criterion,optimizer,epochs=5):
    hist = {'loss':[], 'val_loss': []}
    lowest_loss = 10**5
    
    for epoch in range(epochs):
        loss = 0
        print(f'Epoch: {epoch+1}/{epochs}')
        print('-'*25)
        model.train()
        for x,y in train_loader:
            x = torch.from_numpy(x.numpy()).float().to(device)
            y = torch.from_numpy(y.numpy()).float().to(device)
            optimizer.zero_grad()

            outputs = model(x)

            train_loss = criterion(outputs, y)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        loss = loss / len(train_loader)
        hist['loss'].append(loss)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x,y in test_loader:
                x = torch.from_numpy(x.numpy()).float().to(device)
                y = torch.from_numpy(y.numpy()).float().to(device)
                outputs = model(x)
                val_loss += criterion(outputs, y).item()
        val_loss/=len(test_loader)
        hist['val_loss'].append(val_loss)
        # display the epoch training loss
        print(f"loss = {loss}, val_loss = {val_loss}")
        
        if val_loss<lowest_loss:
            print('One of the best validation accuracy found.\n')
            torch.save(model.state_dict(), 'saved_models/rca_tags_sampled_best.pt')
            lowest_loss = val_loss
    
    torch.save(model.state_dict(), 'saved_models/rca_tags_sampled.pt')
    return hist


# In[23]:


hist = train_model(model,criterion,optimizer,epochs=30)


# In[24]:

plt.clf()
plt.plot(range(len(hist['loss'])), hist['loss'], label = 'Train_loss')
plt.plot(range(len(hist['loss'])), hist['val_loss'], label = 'Validation_loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
plt.clf()

# In[41]:


def get_threshold(model, criterion,train_loader):
    model.eval()
    with torch.no_grad():
        train_loss = []
        for idx,(x,y) in enumerate(train_loader):
            x = torch.from_numpy(x.numpy()).float().to(device)
            y = torch.from_numpy(y.numpy()).float().to(device)
            outputs = model(x)
            loss = criterion(outputs, y).item()
            train_loss.append(loss)
        train_loss = np.array(train_loss)
    threshold = np.mean(train_loss) + np.std(train_loss)
    print('Threshold:', threshold)
    return threshold


# In[48]:


threshold = get_threshold(model, criterion,train_loader)*2


# In[51]:


def predict(model, data, threshold):
    dl = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=2)
    anomalous = []
    model.eval()
    with torch.no_grad():
        for idx,(x,y) in enumerate(train_loader):
            x = torch.from_numpy(x.numpy()).float().to(device)
            y = torch.from_numpy(y.numpy()).float().to(device)
            outputs = model(x)
            test_loss = criterion(outputs, y).item()
            if test_loss>threshold:
                anomalous.append(idx)
    return anomalous


# In[52]:


# if list is empty, there is no anomaly
# number i in anomaly list indicates that data[i:i+32*200] contain anomalous data

anomaly = predict(model, test_data, threshold)
print('test data anomaly',anomaly)





