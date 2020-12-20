
import pandas as pd
import sklearn
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import numpy as np
import fasttext

from lm import train_lm, get_generation_model, gen_positive, gen_negative, model
from removal_bert import train_bert, load_bert, predict_proba_bert, get_proba_bert, replace_with_unk, postoneutral_bert, negtoneutral_bert
from removal_fasttext import predict_proba_fasttext, postoneutral_fasttext, negtoneutral_fasttext, train_fasttext
from tools import manual_seed, load_dataset_rureviews, make_train_for_lm, score_sentences, load_dataset_toxic
from settings import *



manual_seed(1)
print('Reading data')
if data_source == 'rureviews':
    train, test = load_dataset_rureviews(workdir+'women-clothing-accessories.3-class.balanced.csv', sample_size, test_size)
else:
    train, test = load_dataset_toxic(workdir+'toxic.txt', sample_size, test_size)
print(f'Train {len(train)}, test {len(test)}')
print('Classification model:', classification_model)

#bert_model = load_bert(workdir+"sentiment-bert")
print('Start training BERT')
bert_model = train_bert(train, bert_model_id)
result, model_outputs, wrong_predictions = bert_model.eval_model(test[['review','sentiment']], acc=sklearn.metrics.accuracy_score)
print('Classification accuracy of BERT model', (result['tp']+result['tn']) / (len(test)))

if classification_model == 'bert':
    #predict_proba(["Ткань хорошая","не получила посылку, поодовец врёт что заказ придёт, открыла спор деньги не вернули"])

    bert_model.model.config.output_attentions = True
    bert_model.args.silent=True

    print('Prepare positive to neutral')
    neutral_train = postoneutral_bert(train,workdir+f'{model_prefix}pos-to-neutral-train.csv')
    neutral_test = postoneutral_bert(test,workdir+f'{model_prefix}pos-to-neutral-test.csv')
    print(f'Pos to neutral train: {len(neutral_train)}, test: {len(neutral_test)}')

    print('Prepare negative to neutral')
    neutral_train = negtoneutral_bert(train,workdir+f'{model_prefix}neg-to-neutral-train.csv')
    neutral_test = negtoneutral_bert(test,workdir+f'{model_prefix}neg-to-neutral-test.csv')
    print(f'Neg to neutral train: {len(neutral_train)}, test: {len(neutral_test)}')
    #model_prefix = "bert-"
    print('Stage 1 done')
    
elif classification_model == 'fasttext':
    #model_prefix = "ft-"

    print('Prepare train fasttext')
    with open("data.train","w",encoding="utf-8") as fout:
      for i,row in train.iterrows():
        fout.write("__label__" + str(row.sentiment) + "\t" + row.review.replace('\n',' ').replace('\r', '').lower() + "\n")
    with open("data.test","w",encoding="utf-8") as fout:
      for i,row in test.iterrows():
        fout.write("__label__" + str(row.sentiment) + "\t" + row.review.replace('\n',' ').replace('\r', '').lower() + "\n")

    print('Train fasttext')
    train_fasttext()
    #model.predict(["Продавец врёт"])
    #model.predict(["Ткань хорошая"])
    

    print('Prepare positive to neutral')
    neutral_train = postoneutral_fasttext(train)
    pd.DataFrame(neutral_train).to_csv(workdir+f'{model_prefix}pos-to-neutral-train.csv')
    neutral_test = postoneutral_fasttext(test, True)
    pd.DataFrame(neutral_test).to_csv(workdir+f'{model_prefix}pos-to-neutral-test.csv')
    print(f'Pos to neutral train: {len(neutral_train)}, test: {len(neutral_test)}')

    print('Prepare negative to neutral')
    neutral_train = negtoneutral_fasttext(train)
    neutral_test = negtoneutral_fasttext(test)
    pd.DataFrame(neutral_train).to_csv(workdir + f'{model_prefix}neg-to-neutral-train.csv')
    pd.DataFrame(neutral_test).to_csv(workdir + f'{model_prefix}neg-to-neutral-test.csv')
    print(f'Neg to neutral train: {len(neutral_train)}, test: {len(neutral_test)}')

    print('Stage 1 done')
    

print('Make train for LM')
make_train_for_lm()
print('Train LM')
train_lm()
model = get_generation_model()

print('Generate positive from neutral')
positive = gen_positive()
print('Generate negative from neutral')
negative = gen_negative()


print('Calculating score')
positive_score = score_sentences(positive, f'{workdir}{model_prefix}pos-to-neg-test.csv', True)
negative_score = score_sentences(negative, f'{workdir}{model_prefix}neg-to-pos-test.csv', False)

print(f'Overall {classification_model}')
print('Count',len(positive_score['proba']+negative_score['proba']))
print('Accuracy',np.array(positive_score['proba']+negative_score['proba']).mean())
print('BLEU',np.array(positive_score['bleu']+negative_score['bleu']).mean())
print('G-Score',np.array(positive_score['gscore']+negative_score['gscore']).mean())

print('Positive to negative')
print('Count',len(positive_score['proba']))
print('Accuracy',np.array(positive_score['proba']).mean())
print('BLEU',np.array(positive_score['bleu']).mean())
print('G-Score',np.array(positive_score['gscore']).mean())

print('Negative to positive')
print('Count',len(negative_score['proba']))
print('Accuracy',np.array(negative_score['proba']).mean())
print('BLEU',np.array(negative_score['bleu']).mean())
print('G-Score',np.array(negative_score['gscore']).mean())
