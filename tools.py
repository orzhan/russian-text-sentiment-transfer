import os
import torch
import random
import numpy as np
import re

import pandas as pd
import sklearn
from scipy.special import softmax
from sklearn.model_selection import train_test_split

import nltk
from nltk.tokenize import RegexpTokenizer
from settings import *
from removal_bert import get_proba_bert

re_rus = re.compile('[А-Яа-я]')

def manual_seed(seed):
#seed = 42
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)

    
def load_dataset_rureviews(filename, sample_size, test_size):
    dataset = pd.read_csv(filename, sep='\t')
    print('Lines read', len(dataset))
    df_nn = dataset.loc[dataset.sentiment != 'neautral'].copy()
    df_nn.reset_index(inplace=True)

    n_rus = pd.Series([len(re_rus.findall(x)) for x in df_nn.review])

    df = df_nn.loc[n_rus != 0].copy()
    df.reset_index(inplace=True)
    df.drop(columns=['level_0'], inplace=True)
    
    print('Lines filtered',len(df))
    
    if sample_size > 0:
        df = df.sample(sample_size).copy()
        df.reset_index(inplace=True)
        
    print('Will split size',len(df))

    print(df)
    
    df.sentiment =  pd.to_numeric(pd.Series([0 if x == 'negative' else 1 for x in df.sentiment]))

    print(df)
    
    train, test = train_test_split(df, test_size=test_size,stratify=df['sentiment'])
    
    print('Train classes', train.sentiment.value_counts())
    print('Test classes', test.sentiment.value_counts())

    train.drop(columns=['index'], inplace=True)
    test.drop(columns=['index'], inplace=True)
    return train, test
	
def load_dataset_toxic(filename, sample_size, test_size):
    pos = []
    neg = []
    with open(filename, encoding='utf=8') as fin:
        for line in fin:
            labels = line[:line.index(' ')]
            text = line[line.index(' '):]
            if labels == '__label__NORMAL':
                pos.append(text)
            else:
                neg.append(text)
    pos = pd.DataFrame(pos, columns=['review'])
    pos['sentiment'] = 1
    neg = pd.DataFrame(neg, columns=['review'])
    neg['sentiment'] = 0

    df = pd.concat([pos.sample(len(neg)),neg])
    print('Lines filtered',len(df))

    if sample_size > 0:
        df = df.sample(sample_size).copy()
        df.reset_index(inplace=True)

    print('Will split size',len(df))

    print(df)

    train, test = train_test_split(df, test_size=test_size,stratify=df['sentiment'])

    print('Train classes', train.sentiment.value_counts())
    print('Test classes', test.sentiment.value_counts())

    #train.drop(columns=['index'], inplace=True)
    #test.drop(columns=['index'], inplace=True)

    return train, test

def make_train_for_lm():
    df = pd.read_csv(f'{workdir}{model_prefix}neg-to-neutral-train.csv')
    train=[]
    for i, row in df.iterrows():
      try:
        s = "StartPositive " + row[1] + " ToNegative " + row[2] + " End "
        s = s.replace('[CLS] ','').replace('[SEP]', '').replace('\n',' ').replace('\r','')
        if remove_unk:
          s = s.replace('<unk>', '').replace('  ',' ')
        train.append(s)
      except Exception as ex:
        print(ex)
        break
  
    df = pd.read_csv(f'{workdir}{model_prefix}pos-to-neutral-train.csv')

    for i, row in df.iterrows():
      try:
        s = "StartNegative " + row[1] + " ToPositive " + row[2] + " End "
        s = s.replace('[CLS] ','').replace('[SEP]', '').replace('\n',' ').replace('\r','')
        if remove_unk:
          s = s.replace('<unk>', '').replace('  ',' ')
        train.append(s)
      except Exception as ex:
        print(ex)
        break

    with open("train.txt", "w", encoding='utf-8') as fout:
      fout.write("\n".join(train))

from collections import Counter
import math
import numpy as np

def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return bleu(stats)
    

def score_sentences(gen, filename, is_positive):
    df = pd.read_csv(filename)
    probas = []
    bleus = []
    gscores = []
    for i, row in df.iterrows():
      gen_new = row['0']
      try:
        if is_positive:
            prev_proba = 1 - get_proba_bert(row['1'])
            proba = 1 - get_proba_bert(gen_new)
        else:
            prev_proba = get_proba_bert(row['1'])
            proba = get_proba_bert(gen_new)
        bleu = get_bleu([row['1']],[gen_new])
        probas.append(proba)
        bleus.append(bleu)
        #gscores.append(proba * bleu)
      except Exception as ex:
        print(ex)
    accuracy = np.sum(np.array(probas) >= 0.5, axis=0) / len(probas)
    gscore = accuracy * np.array(bleus).mean()
    return {'proba': probas, 'bleu': bleus, 'accuracy': accuracy, 'bleu_mean': np.array(bleus).mean(), 'gscore': gscore}
    