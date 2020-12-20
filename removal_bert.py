import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import numpy as np
import re
from settings import *
from scipy.special import softmax

bert_model = None
re_rus = re.compile('[А-Яа-я]')

def train_bert(train, bert_model_id):
    global bert_model
    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.num_train_epochs = 1
    model_args.learning_rate = 2e-5
    model_args.output_dir = 'bert_model'
    model = ClassificationModel(
        "bert", bert_model_id, args=model_args
    )

    train_df =  train[['review','sentiment']]

    print(train_df)

    model.train_model(train_df, acc=sklearn.metrics.accuracy_score)
    bert_model = model
    return model
    
def load_bert(filename):
    global bert_model
    model_args = ClassificationArgs()
    model_args.overwrite_output_dir = True
    model_args.reprocess_input_data = True
    model_args.num_train_epochs = 1
    model_args.learning_rate = 2e-5
    bert_model = ClassificationModel("bert",filename, args=model_args)
    return bert_model
    
def predict_proba_bert(texts):
  #model.args.silent=True
  predictions, raw_outputs = bert_model.predict(texts)
  probabilities = softmax(raw_outputs, axis=1)
  return [p[1] for p in probabilities]

def get_proba_bert(text):
  inputs = bert_model.tokenizer(text, return_tensors='pt', add_special_tokens=True)
  token_type_ids = inputs['token_type_ids'].to(torch.device("cuda"))
  input_ids = inputs['input_ids'].to(torch.device("cuda"))
  logits = bert_model.model(input_ids, token_type_ids=token_type_ids)[0].detach().cpu().numpy()
  probabilities = softmax(logits, axis=1)
  return probabilities[0][1]
  

def average_last_layer_by_head(attentions):
    last_multihead_attn = attentions[-1]
    # For each multihead attention, get the attention weights going into the CLS token
    cls_attn = last_multihead_attn[:, :, 0, :]
    # Average across attention heads
    cls_attn = torch.mean(cls_attn, axis=1)
    # Normalize across tokens
    total_weights = torch.sum(cls_attn, axis=-1, keepdims=True)
    norm_cls_attn = cls_attn / total_weights
    return norm_cls_attn.detach().cpu().numpy().tolist()

def merge_wordpiece_tokens(paired_tokens):
    new_paired_tokens = []
    n_tokens = len(paired_tokens)
    i = 0
    while i < n_tokens:
        current_token, current_weight = paired_tokens[i]
        if current_token.startswith('##'):
            previous_token, previous_weight = new_paired_tokens.pop()
            merged_token = previous_token
            merged_weight = [previous_weight]
            while current_token.startswith('##'):
                merged_token = merged_token + current_token.replace('##', '')
                merged_weight.append(current_weight)
                i = i + 1
                current_token, current_weight = paired_tokens[i]
            merged_weight = np.mean(merged_weight)
            new_paired_tokens.append((merged_token, merged_weight))
        else:
            new_paired_tokens.append((current_token, current_weight))
            i = i + 1
    return new_paired_tokens

def get_att(text):
  inputs = bert_model.tokenizer(text, return_tensors='pt', add_special_tokens=True)
  token_type_ids = inputs['token_type_ids'].to(torch.device("cuda"))
  input_ids = inputs['input_ids'].to(torch.device("cuda"))
  attention = bert_model.model(input_ids, token_type_ids=token_type_ids)[-1]
  input_id_list = input_ids[0].tolist() # Batch index 0
  tokens = bert_model.tokenizer.convert_ids_to_tokens(input_id_list)
  return merge_wordpiece_tokens(list(zip(tokens, average_last_layer_by_head(attention)[0])))


def replace_with_unk(text, alpha, beta):
  att = get_att(text)
  tokens_att = []
  for i,a in enumerate(att):
    tokens_att.append([a[0], a[1], i])
  #print(tokens_att)
  # get top attention tokens with letters
  tokens_sorted = sorted(tokens_att, key=lambda tup: -tup[1])
  proba0 = get_proba_bert(text)
  if proba0>alpha:
    dir='down'
  elif proba0<1-alpha:
    dir='up'
  else:
    return text

  alpha_token_count = sum([1 if re_rus.search(token[0]) else 0 for token in tokens_sorted])
  #print('alpha_token_count',alpha_token_count)

  modified_text = text
  # take tokens with attention > alpha%, no more than beta % of sentence
  removed = 0
  for tok in tokens_sorted:
    token, att, index = tok
    if att < 0.005:
      break
    if re_rus.search(token):
      removed += 1
      #print('remove', tokens_att[index][0], removed, beta * len(tokens_att))
      tokens_att[index][0] = '<unk>'
      if removed >= beta * alpha_token_count:
        break
      modified_text = bert_model.tokenizer.convert_tokens_to_string([x[0] for x in tokens_att])
      if dir == 'down':
        if get_proba_bert(modified_text) < alpha:
          break
      if dir == 'up':
        if get_proba_bert(modified_text) > alpha:
          break
  #modified_text = bert_model.tokenizer.convert_tokens_to_string([x[0] for x in tokens_att])

  return modified_text

def postoneutral_bert(dataset, filename):
  alpha = alpha_bert
  beta = beta_bert
  neutral = []
  c = 0
  for i, row in dataset.iterrows():
    if row['sentiment'] == 0:
      continue
    c += 1
    if c % 1000 == 0:
      print(c, len(dataset))
      pd.DataFrame(neutral).to_csv(filename)
    text = row['review']
    try:
      new_text = replace_with_unk(text, alpha, beta)
      #print(new_text)
      #print(predict_proba([text, new_text]))
      neutral.append((new_text,text))
    except Exception as ex:
      print(ex)
  pd.DataFrame(neutral).to_csv(filename)
  return neutral

def negtoneutral_bert(dataset, filename):
  alpha = alpha_bert
  beta = beta_bert
  neutral = []
  c = 0
  for i, row in dataset.iterrows():
    if row['sentiment'] == 1:
      continue
    c += 1
    if c % 5000 == 0:
      print(c, len(dataset))
      pd.DataFrame(neutral).to_csv(filename)
    text = row['review']
    try:
      new_text = replace_with_unk(text, alpha, beta)
      neutral.append((new_text,text))
    except Exception as ex:
      print(ex)
  pd.DataFrame(neutral).to_csv(filename)
  return neutral