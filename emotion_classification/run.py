#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import os
import sys
import argparse
import random


import numpy as np
import tqdm
import pickle
import pandas as pd
from matplotlib import pyplot as plt

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")


# In[2]:


def logging_storage(logfile_path):
    logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
    logging.info(torch.__version__)
    logging.info(device)

parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=2e-5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
parser.add_argument('--ts', help='Test Size', default=0.2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)



args = parser.parse_args()



lr = args.lr
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
test_size = args.ts
model = 'xlnet'
num_labels = 4
denom = args.adaptive


# In[4]:


ending_path = ('%s_%d_bs_%d_adamw_data_%d_lr_%s_%d' %(model, MAX_LEN, batch_size,(1 - test_size)*100, str(lr).replace("-",""),denom))
ending_path


# In[5]:


save_model_path = "../models/" + ending_path
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if not os.path.exists("../logs/"):
    os.mkdir("../logs/")
logfile_path = "../logs/" + ending_path


# In[6]:


logging_storage(logfile_path)


# ## Data Loading

# In[7]:


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

df = pd.read_csv('../data/Moody_Lyrics.csv', index_col = 0)

# In[10]:

songs = df['Lyrics'].tolist()

# In[11]:


tokenized_texts = [tokenizer.tokenize(song) for song in songs]


# In[12]:


input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]


# In[13]:


input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[14]:


# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


# In[16]:
### Angry = 0, Happy = 1, Relaxed = 2, Sad = 3


output_moods = df['Mood_Numeric'].tolist()



# ## Process ML
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, output_moods, 
                                                            random_state=2018, test_size=test_size)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=test_size)


# In[18]:


train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)



# In[20]:


train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# In[21]:


model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
model = nn.DataParallel(model)
model.to(device)

logging.info("Model Loaded!")

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters,
                     lr=lr)

# ## Helper functions

# In[42]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy() 
    return np.sum(pred_flat == labels_flat), pred_flat


# In[43]:


def train(i):
    model.train()
    total_loss = 0.0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    train_len = 0
    f_acc = 0
    
    ## adaptive lr
    optimizer.param_groups[0]['lr'] *= (0.1)**(1/denom)
    
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        if b_labels.size(0) <= 1:
            continue

        
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        pred = outputs[1].detach().cpu().numpy()
        batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
        f_acc += batch_f_acc
        loss = outputs[0]
        loss.sum().backward()
        optimizer.step()
        
        
        labels_flat = b_labels.flatten().cpu().detach().numpy()
        total_actual_label = np.concatenate((total_actual_label, labels_flat))
        total_predicted_label = np.concatenate((total_predicted_label, pred_flat))
        
        total_loss += outputs[0].sum()
        train_len += b_input_ids.size(0)
        
        if step%100 == 0 and step:
            precision, recall, f1_measure, _ =  precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
            logging.info("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %(train_len*100.0/train_inputs.size(0), i, step,total_loss/train_len, f_acc*100.0/train_len,precision*100., recall*100., f1_measure*100.))

        # if torch.cuda.device_count() > 1:
        #     p = 100
        #     path = save_model_path + '/e_' + str(i) + "_" + str(p) + ".ckpt"
        #     torch.save(model.module.state_dict(), path)
        # else:
        #     torch.save(model.state_dict(), path)

    precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
    logging.info("Train: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %(train_len*100.0/train_inputs.size(0), i, step,total_loss/train_len, f_acc*100.0/train_len, precision*100., recall*100., f1_measure*100.))

# In[44]:
def eval(i):
    model.eval()
    val_len = 0
    total_loss = 0
    total_predicted_label = np.array([])
    total_actual_label = np.array([])
    f_acc = 0
    
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader):
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            batch_f_acc, pred_flat = flat_accuracy(pred, b_labels)
            f_acc += batch_f_acc
            
            labels_flat = b_labels.flatten().cpu().detach().numpy()
            total_actual_label = np.concatenate((total_actual_label, labels_flat))
            total_predicted_label = np.concatenate((total_predicted_label, pred_flat))
            
            val_len += b_input_ids.size(0)
            total_loss += outputs[0].sum()

        if step%100 == 0 and step:
            precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
            logging.info("Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %(val_len*100.0/validation_inputs.size(0), i, step,total_loss/val_len, f_acc*100.0/val_len,precision*100., recall*100., f1_measure*100.))
    
    precision, recall, f1_measure, _ = precision_recall_fscore_support(total_actual_label, total_predicted_label, average='macro')
    logging.info("Eval: %5.1f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.3f\tPrecision= %5.3f\tRecall= %5.3f\tF1_score= %5.3f" %(val_len*100.0/validation_inputs.size(0), i, step,total_loss/val_len, f_acc*100.0/val_len,precision*100., recall*100., f1_measure*100.))
        


# In[ ]:


for i in range(num_epochs):
    train(i)
    eval(i)

