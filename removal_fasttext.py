import pandas as pd
import numpy as np
import fasttext
from settings import *

ft_model = None

def train_fasttext():
    global ft_model
    ft_model = fasttext.train_supervised(input="data.train", lr=0.1, epoch=5)
    ft_model.save_model("fast.bin")
    print('train_fasttext', ft_model)
    print(ft_model.test("data.test"))
    return ft_model

  
def predict_proba_fasttext(texts):
  values = ft_model.predict([s.replace('\n',' ').replace('\r','') for s in texts])
  #print(values)
  proba = []
  for i, v in enumerate(values[0]):
    val = values[1][i][0] if v[0] == '__label__1' else 1-values[1][i][0]
    proba.append(val)
  return proba

def postoneutral_fasttext(dataset, return_all = False):
  alpha=alpha_fasttext     # target polarity of neutral sentence
  beta=beta_fasttext      # minimum amount of remaining words

  neutral = []
  c = 0
  for i, row in dataset.iterrows():
    if row['sentiment'] == 0:
      continue
    c += 1
    if c % 5000 == 0:
      print(c, len(dataset))
    text = row['review']
    orig = text
    prob = predict_proba_fasttext([text])[0]
    #print('---------')
    #print(prob, text)
    if prob < alpha:
      if return_all:
        neutral.append((text, orig))
      continue
    n_words = len(text.split(' '))
    n_deleted = 0
    while prob > alpha:
      toks = text.split(' ')
      max_tok = None
      max_tok_val = prob
      max_tok_index = None
      text_parts = []
      for t in toks:
        text_parts.append(' '.join([tt if tt != t else "<unk>" for tt in toks ]))
      probabilities = predict_proba_fasttext(text_parts)
      #print('probabilities',probabilities)
      for index, prob in enumerate(probabilities):
        #print('prob', prob, 'max_tok_val', max_tok_val, 'prob[1]', prob[1])
        if prob<max_tok_val:
          max_tok_val=prob
          max_tok=toks[index]
          max_tok_index = index
      n_deleted += 1
      if n_deleted/n_words > 1-beta or max_tok_index is None:
        #text = None
        break
      #print(max_tok_val, max_tok)
      prob = max_tok_val
      text = text_parts[max_tok_index]
      #print(text)
    if text is not None:
      neutral.append((text, orig))
    elif return_all:
      neutral.append((orig, orig))
        #print(text_part)
  return neutral

def negtoneutral_fasttext(dataset, return_all = False):
  alpha=alpha_fasttext     # target polarity of neutral sentence
  beta=beta_fasttext      # minimum amount of remaining words

  neutral = []
  c = 0
  for i, row in dataset.iterrows():
    if row['sentiment'] == 1:
      continue
    c += 1
    if c % 5000 == 0:
      print(c, len(dataset))
    text = row['review']
    orig = text
    prob = predict_proba_fasttext([text])[0]
    #print('---------')
    #print(prob, text)
    if prob > 1-alpha:
      if return_all:
        neutral.append((text, orig))
      continue
    n_words = len(text.split(' '))
    n_deleted = 0
    while prob < 1-alpha:
      toks = text.split(' ')
      max_tok = None
      max_tok_val = 0
      max_tok_index = None
      text_parts = []
      for t in toks:
        text_parts.append(' '.join([tt if tt != t else "<unk>" for tt in toks ]))
      probabilities = predict_proba_fasttext(text_parts)
      #print('probabilities',probabilities)
      for index, prob in enumerate(probabilities):
        #print('prob', prob, 'max_tok_val', max_tok_val, 'prob[1]', prob[1])
        if prob>max_tok_val:
          max_tok_val=prob
          max_tok=toks[index]
          max_tok_index = index
      n_deleted += 1
      if n_deleted/n_words > 1-beta or max_tok_index is None:
        #text = None
        break
      #print(max_tok_val, max_tok)
      prob = max_tok_val
      text = text_parts[max_tok_index]
      #print(text)
    if text is not None:
      neutral.append((text, orig))
    elif return_all:
       neutral.append((text, orig))
        #print(text_part)
  return neutral
