# Settings

#!git clone https://github.com/sismetanin/rureviews.git
#!mv rureviews/*.csv ./

#workdir = '/content/gdrive/My Drive/'
workdir = './'
alpha_bert = 0.99
beta_bert = 0.6

alpha_fasttext = 0.55
beta_fasttext = 0.6

remove_unk = False
sample_size = 2000
test_size = 20

max_generate=10

bert_model_id = "DeepPavlov/rubert-base-cased-sentence"
gpt_model_id = "sberbank-ai/rugpt3small_based_on_gpt2"

data_source='toxic'

classification_model='fasttext'
model_prefix='ft'