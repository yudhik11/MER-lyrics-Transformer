import logging
import os
import sys
import argparse
import random
import logging

import numpy as np
import tqdm
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, f1_score

import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import XLNetTokenizer, XLNetForSequenceClassification, XLNetModel, AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")

class multitask_model(nn.Module):
    def __init__(self, transformer):
        super(multitask_model, self).__init__()
        self.transformer = transformer
        self.fc_middle = nn.Sequential(nn.ReLU(),
                                nn.Linear(8, 8),
                                 nn.Dropout(0.25)
                                )
        self.fc_quadrant = nn.Sequential(
                                nn.ReLU(inplace=True),
                                nn.Linear(8, 4),
#                                  nn.Dropout(0.25)
                                )
        self.fc_valence = nn.Sequential(nn.Linear(8, 2),
#                                  nn.Dropout(0.25)
                                )
        self.fc_arousal = nn.Sequential(nn.Linear(8, 2),
#                                  nn.Dropout(0.25)
                                )
    def forward(self, b_input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.transformer(b_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)            
        output = self.fc_middle(outputs[1])
        fc_quadrant_out = self.fc_quadrant(output)
        fc_valence_out = self.fc_valence(output)
        fc_arousal_out = self.fc_arousal(output)
        return torch.cat((fc_quadrant_out, fc_valence_out, fc_arousal_out), axis = 1)

xlnet_transformer = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=8)
# xlnet_transformer = XLNetModel.from_pretrained('xlnet-base-cased')
model = multitask_model(xlnet_transformer)
model = nn.DataParallel(model)
model.to(device)