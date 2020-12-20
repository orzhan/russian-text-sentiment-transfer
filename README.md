# Intro

Unsupervised sentiment transfer is changing the sentiment of the given text while preserving its original content without using pairs of parallel sentences. Sentiment transfer is a particular case of a wider problem - text style transfer. One of the most popular approaches to sentiment transfer is removing original sentiment markers (words specific to a sentiment) from a sentence and then adding target sentiment markers[Li et al., 2018] [Lee, 2020]. Proposed model is built based on this approach and makes use of a pre-trained language model [Brown et al., 2020] to generate fluent sentences. The model works with texts written in Russian.

# Datasets

RuReviews: An Automatically Annotated Sentiment Analysis Dataset for Product Reviews in Russian: https://github.com/sismetanin/rureviews
Toxic Russian Comments: https://www.kaggle.com/alexandersemiletov/toxic-russian-comments

# Training

Edit settings.py (`data_source` may be 'toxic' or 'rureviews', `classification_model` may be 'fasttext' or 'bert'). 

Run `python sentiment_transfer.py`

# Inference

Fine-tuned models are available on [https://huggingface.co/](https://huggingface.co/orzhan). You can use this [colab notebook](https://www.google.com) to run them.

# Examples for RuReviews dataset

**Positive to negative**

Кофта приятная, цвет супер → Кофта приятная, пришла с дырой

товар соответствует описанию в Ростов пришёл за 3 недели → Товар не соответствует описанию в Ростов пришёл не 3 недели

**Negative to positive**

Размер вообще не совпадает, на куклу одежда, даже на S размер не тянет → Все как в описании, все очень хорошо совпадает, на русский s одежда

кофточка пришла не по размеру заказывала хl а как будто s → кофта пришла по размеру подошла хl а как просила s