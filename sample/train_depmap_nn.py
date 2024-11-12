#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from datetime import datetime
import os
import gc
import numpy as np
sys.path.append('../code')


# In[2]:


from torch.utils.data import DataLoader
import torch
from torch import nn
from util import pearson_corr


# In[3]:


from depmap_NN import DepMapPairsDataset, DepMapPairsEncoder


# In[4]:


# run options
torch.manual_seed(42)
num_workers = 4
num_epochs = 5
batch_size = 1000
learning_rate = 1e-3
use_GPU = True
if not torch.cuda.is_available():
    warn('torch.cuda.is_available() returned False, using CPU')
else:
    print('using GPU')

device = torch.device("cuda:0" if use_GPU and torch.cuda.is_available() else "cpu")


# In[5]:


inp_dir = '../data/preproc/depmap_all_dtypes_processed'
X_RNA_file = os.path.join(inp_dir, 'X_RNA.csv')
X_SNV_file = os.path.join(inp_dir, 'X_SNV.csv')
Y_CRISPR_file = os.path.join(inp_dir, 'X_CRISPRGeneEffect.csv')
X_ppis_file = os.path.join(inp_dir, 'X_ppi.csv')


# In[6]:


output_dir = '../data/preproc/depmap_sandbox_models'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# In[7]:


train_idx_file = os.path.join(inp_dir, 'train_idx.csv')
test_idx_file = os.path.join(inp_dir, 'test_idx.csv')


# In[8]:


train_dataset = DepMapPairsDataset(omics_matrix_files=[X_RNA_file, X_SNV_file], 
                                   ppis_file=X_ppis_file,
                                    crispr_score_file=Y_CRISPR_file, 
                                    index_file=train_idx_file)


# In[9]:


validation_dataset = DepMapPairsDataset(omics_matrix_files=[X_RNA_file, X_SNV_file], 
                                       ppis_file=X_ppis_file,
                                        crispr_score_file=Y_CRISPR_file, 
                                        index_file=train_idx_file)


# In[10]:


model = DepMapPairsEncoder(train_dataset.omics_matrix_idx, train_dataset.ppi_matrix_idx)


# In[11]:


model.omics_matrix_idx


# In[12]:


model


# In[13]:


model.to(device)


# In[14]:


train_dataset.index = train_dataset.index[0:24000]


# In[15]:


validation_dataset.index = validation_dataset.index[24000:32000]


# In[16]:


# we make up for the inefficiency on loading the data by parallelizing things. 
DL_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
DL_val = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[17]:


print(datetime.now())


# In[18]:


optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
loss_fun = nn.MSELoss()
train_loss_avg = []
train_corr_avg = []
print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    train_corr_avg.append(0)
    num_batches = 0
    print("Epoch {}".format(epoch))
    print(datetime.now())
    for batch, target in DL_train:
        x = batch.to(device)
        y = target.to(device)
        pred = model.forward(x)
        # reconstruction error
        loss = loss_fun(pred, y)
        # print('Batch loss:{}'.format(loss.item()))
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # one step of the optmizer (using the gradients from backpropagation)
        optimizer.step()
        
        train_loss_avg[-1] += loss.item()
        train_corr_avg[-1] += pearson_corr(y, pred)
        num_batches += 1
        
    torch.save(model, output_dir + '/model_' + str(epoch) + '.pt')
    train_loss_avg[-1] /= num_batches
    train_corr_avg[-1] /= num_batches
    print('Epoch [%d / %d] Train Loss: %f Train Correlation: %f' % (epoch+1, num_epochs, train_loss_avg[-1], train_corr_avg[-1]))
    print(datetime.now())
    # do validation
    num_batches = 0
    val_loss_avg = []
    val_corr_avg = []
    for batch, target in DL_val:
        val_loss_avg.append(0)
        val_corr_avg.append(0)
        x = batch.to(device)
        y = target.to(device)
        pred = model.forward(x)
        # reconstruction error
        loss = loss_fun(pred, y)
        val_loss_avg[-1] += loss.item()
        val_corr_avg[-1] += pearson_corr(y, pred)
        num_batches += 1
    val_loss_avg[-1] /= num_batches
    val_corr_avg[-1] /= num_batches
    print('Epoch [%d / %d] Val Loss: %f Val Correlation: %f' % (epoch+1, num_epochs, val_loss_avg[-1], val_corr_avg[-1]))
    print(datetime.now())
    gc.collect()


# In[ ]:




