#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

# load training data
train = pd.read_csv('BERT_proj/train_E6oV3lV.csv', encoding='iso-8859-1')
train.shape

import re

# clean text from noise
def clean_text(text):
    # filter to allow only alphabets
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    
    # remove Unicode characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # convert to lowercase to maintain consistency
    text = text.lower()
       
    return text

train['clean_text'] = train.tweet.apply(clean_text)

from sklearn.model_selection import train_test_split

# split into training and validation sets
X_tr, X_val, y_tr, y_val = train_test_split(train.clean_text, train.label, test_size=0.25, random_state=42)

print('X_tr shape:',X_tr.shape)




from bert_serving.client import BertClient

# make a connection with the BERT server using it's ip address
bc = BertClient(ip="YOUR_SERVER_IP")
# get the embedding for train and val sets
X_tr_bert = bc.encode(X_tr.tolist())
X_val_bert = bc.encode(X_val.tolist())





from sklearn.linear_model import LogisticRegression

# LR model
model_bert = LogisticRegression()
# train
model_bert = model_bert.fit(X_tr_bert, y_tr)
# predict
pred_bert = model_bert.predict(X_val_bert)





from sklearn.metrics import accuracy_score

print(accuracy_score(y_val, pred_bert))







