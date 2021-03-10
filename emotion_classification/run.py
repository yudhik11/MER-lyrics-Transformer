#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import logging
import os
import sys
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
import tqdm
import pickle
from matplotlib import pyplot as plt


# In[2]:


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


# In[3]:


from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW


# In[4]:

import argparse

import pandas as pd


# In[5]:


df = pd.read_csv('../data/Moody_Lyrics.csv', index_col = 0)
df.index
df.columns


# In[6]:


### Angry = 0, Happy = 1, Relaxed = 2, Sad = 3


# In[7]:


parser = argparse.ArgumentParser()
parser.add_argument('--lr', help='Learning Rate', default=2e-5, type=float)
parser.add_argument('--epochs', help='Number of Epochs', default=20, type=int)
parser.add_argument('--ml', help='Max Len of Sequence', default=1024, type=int)
parser.add_argument('--bs', help='Batch Size', default=8, type=int)
parser.add_argument('--ts', help='Test Size', default=0.2, type=float)
parser.add_argument('--adaptive', help='Adaptive LR', default='20', type=float)

args = parser.parse_args()

print(args)
## GENERAL Prameters
lr = args.lr
num_epochs = args.epochs
MAX_LEN = args.ml
batch_size = args.bs
test_size = args.ts
model = 'xlnet'
num_labels = 4
denom = args.adaptive

ending_path = ('%s_%d_bs_%d_adamw_data_%d_lr_%s_%d' %(model, MAX_LEN, batch_size,(1 - test_size)*100, str(lr).replace("-",""),denom))


save_model_path = "../models/" + ending_path
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if not os.path.exists("../logs/"):
    os.mkdir("../logs/")
logfile_path = "../logs/" + ending_path

# for key in logging.Logger.manager.loggerDict:
#     logging.getLogger(key).setLevel(logging.CRITICAL)
    
logging.basicConfig(filename=logfile_path, filemode='a', level=logging.INFO, format='%(asctime)s => %(message)s')
logging.info(torch.__version__)
logging.info(device)


# In[9]:


tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)


# In[10]:

songs = df['Lyrics'].tolist()

# In[11]:


tokenized_texts = [tokenizer.tokenize(song) for song in songs]


# In[12]:


input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]


# In[13]:


input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")


# In[14]:


input_ids


# In[15]:


# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)


# In[16]:
### Angry = 0, Happy = 1, Relaxed = 2, Sad = 3


output_moods = df['Mood_Numeric'].tolist()


# In[17]:


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


# In[22]:


logging.info("Model Loaded!")


# In[23]:


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


# In[24]:


def train(i, denom):
    model.train()
    total_loss = 0.0
    train_len = 0
    f_acc = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*(0.1**(1/denom))
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        pred = outputs[1].detach().cpu().numpy()
        f_acc += flat_accuracy(pred, b_labels)

        loss = outputs[0]
        loss.sum().backward()
        total_loss += outputs[0].sum()
        train_len += b_input_ids.size(0)
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        if step%100 == 0 and step:
            logging.info("Train: %5.5f\tEpoch: %d\tIter: %5.5f\tLoss: %5.5f\tAcc= %5.5f" %(train_len*100.0/train_inputs.size(0),i,step*(batch_size)*100.0/(train_inputs.size(0)),total_loss/train_len,f_acc*100.0/train_len))

#    if (i + 1)%5 == 0 and i > 0:
#        if torch.cuda.device_count() > 1:
#            p = int(train_len*100.0/train_inputs.size(0))
#            path = save_model_path + '/e_' + str(i) + "_" + str(p) + ".ckpt"
#            torch.save(model.module.state_dict(), path)
#        else:
#            torch.save(model.state_dict(), path)

    return train_len*100.0/train_inputs.size(0), i, 100.0, total_loss/train_len, f_acc*100.0/train_len




def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    labels_flat = labels_flat.cpu().detach().numpy() 
    return np.sum(pred_flat == labels_flat)




def eval(i):
    model.eval()
    val_len = 0
    total_loss = 0
    f_acc = 0
    with torch.no_grad():
        for step, batch in enumerate(validation_dataloader):
            batch = tuple(t.cuda() for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            pred = outputs[1].detach().cpu().numpy()
            f_acc += flat_accuracy(pred, b_labels)
            val_len += b_input_ids.size(0)
            total_loss += outputs[0].sum()
            
        if step%100 == 0 and step:
            logging.info("Eval: %5.5f\tEpoch: %d\tIter: %5.5f\tLoss : %5.5f\tAcc: %5.5f" %(val_len*100.0/validation_inputs.size(0),i,step*(batch_size)*100.0/(validation_inputs.size(0)),total_loss/val_len,f_acc*100.0/val_len))
            
    return (val_len*100.0/validation_inputs.size(0),i,100.0,total_loss/val_len,f_acc*100.0/val_len)



logging.info("Training Started!")

for i in range(num_epochs):
    percent_done,epoch,Iter,loss,f_acc = train(i,denom)
    logging.info("Train: %5.5f\tEpoch: %d\tIter: %d\tLoss: %5.5f\tAcc= %5.5f" %(percent_done,epoch,Iter,loss,f_acc))
    percent_done,epoch,Iter,loss,f_acc = eval(i)
    logging.info("Eval: %5.5f\tEpoch: %d\tIter: %d\tLoss : %5.5f\tAcc: %5.5f" %(percent_done,epoch,Iter,loss,f_acc))

