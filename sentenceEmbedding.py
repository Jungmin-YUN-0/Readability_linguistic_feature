import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sentence_transformers import SentenceTransformer
import re
import itertools
import numpy as np
import pandas as pd
import dill


#=================================# dataset 준비 #=================================#
train = pd.read_csv(r'./onestopeng/train.csv')
test = pd.read_csv(r'./onestopeng/test.csv')


#==============================# Sentence Embedding #==============================#
bert_transformers = SentenceTransformer('bert-base-nli-mean-tokens')

# sentence Bert에 mean pooling 한번 더 적용
def mean(z): 
    return sum(itertools.chain(z))/len(z)

# sentence Bert
def embeddToBERT(text):
    sentences = re.split('!|\?|\.',text)
    sentences = list(filter(None, sentences)) 
    result = bert_transformers.encode(sentences)
    feature = [mean(x) for x in zip(*result)]
    return feature


print('[Info] Start Embedding...')
bert_word_training_features = train['text'].apply(embeddToBERT)
bert_word_test_features = test['text'].apply(embeddToBERT)

feature_train = [x for x in bert_word_training_features.transpose()]
#bert_word_training_features = np.asarray(feature_train)

feature_test = [x for x in bert_word_test_features.transpose()]
#bert_word_test_features = np.asarray(feature_test)
#==================================================================================#

data = {
    'features' : {'f_train':feature_train, 'f_test':feature_test},
    'labels' : {'l_train':train['label'].tolist(), 'l_test':test['label'].tolist()}
    }
print('[Info] Saving Pickle...')

dill.dump(data, open("sentenceEmbedding.pickle", 'wb'))
print('[Info] Done...')
