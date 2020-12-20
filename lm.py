import pandas as pd
from settings import *

from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)
from simpletransformers.language_generation import LanguageGenerationModel
from simpletransformers.config.model_args import LanguageGenerationArgs
from removal_bert import get_proba_bert

model = None

def train_lm():
    model_args = LanguageModelingArgs()
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_train_epochs = 1
    model_args.dataset_type = "simple"
    model_args.mlm = False  # mlm must be False for CLM
    #model.max_seq_length = 512
    #model.block_size = 512

    train_file = "train.txt"
    #test_file = "data/test.txt"

    model = LanguageModelingModel("gpt2", gpt_model_id, args=model_args)

    # Train the model
    model.train_model(train_file) #, eval_file=test_file
    
def get_generation_model():
    global model
    model_args = LanguageGenerationArgs()
    model_args.max_length = 200
    model_args.top_k = 0
    model_args.top_p = 0.9
    #model_args.top_p = 0.9
    model_args.stop_token = "End"

    model = LanguageGenerationModel("gpt2", "outputs", args=model_args)
    return model
    
def gen_positive():
    df = pd.read_csv(f'{workdir}{model_prefix}pos-to-neutral-test.csv')
    gen = []
    for i, row in df.iterrows():
    #print(df.iloc[i]['1'])
      if i % 50 == 0:
        print(i, len(df))
        pd.DataFrame(gen).to_csv(f'{workdir}{model_prefix}pos-to-neg-test.csv')
      if i > max_generate:
        break
      s = row['0'].replace('\n',' ').replace('\r','')
      if remove_unk:
          s = s.replace('<unk>', '').replace('  ', ' ')
      ss = model.generate("StartPositive" + s + " ToNegative ", {'max_length':200})[0]
      ss = ss[ss.index('ToNegative')+len('ToNegative'):].strip()
      gen_new = ss.replace('\n', ' ').replace('\r', '').replace('<unk>', '')
      if ss.find('End') != -1:
        gen_new = ss[:ss.index('End')]
      gen.append((gen_new, row['1']))
    pd.DataFrame(gen).to_csv(f'{workdir}{model_prefix}pos-to-neg-test.csv')
    return gen
    
def gen_negative():
    df = pd.read_csv(f'{workdir}{model_prefix}neg-to-neutral-test.csv')
    gen = []
    for i, row in df.iterrows():
    #print(df.iloc[i]['1'])
      if i % 50 == 0:
        print(i, len(df))
        pd.DataFrame(gen).to_csv(f'{workdir}{model_prefix}neg-to-pos-test.csv')
      if i > max_generate:
        break
      s = row['0'].replace('\n',' ').replace('\r','')
      if remove_unk:
          s = s.replace('<unk>', '').replace('  ', ' ')
      ss = model.generate("StartNegative" + s + " ToPositive ", {'max_length':200})[0]
      ss = ss[ss.index('ToPositive')+len('ToPositive'):].strip()
      gen_new = ss.replace('\n', ' ').replace('\r', '').replace('<unk>', '')
      if ss.find('End') != -1:
        gen_new = ss[:ss.index('End')]
      gen.append((gen_new, row['1']))
    pd.DataFrame(gen).to_csv(f'{workdir}{model_prefix}neg-to-pos-test.csv')
    return gen
    