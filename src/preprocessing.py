import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences
from transformers import DistilBertTokenizer

import os


def rnn_preprocess(sentence):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    punct = string.punctuation
    
    sentence = sentence.lower()
    sentence = re.sub(r'\W+', ' ', sentence)
    clean_sentence = []
    
    for word in re.split(r'(\W+)', sentence):
        if word not in stopwords_en:
            stem_word = stemmer.stem(word)
            clean_sentence.append(stem_word)

    clean_sentence = ''.join(clean_sentence)
    clean_sentence = re.sub(f'r[{re.escape(punct)}]', ' ', clean_sentence)
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
    
    return clean_sentence


def rnn_tokenizer(corpus, max_len=None):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    vectorize_layer = tf.keras.layers.TextVectorization(output_sequence_length=max_len)
    vectorize_layer.adapt(corpus)

    return vectorize_layer


if __name__ == '__main__':
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data'
        )
    )
    
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))

    MAX_TOKENS = None
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 128

    comics_data['description'] = comics_data['description'].map(rnn_preprocess)
    comics_data = comics_data.drop_duplicates('description')
    
    comics_corpus = comics_data[['description', 'y']].copy()
    comics_corpus['y'] = comics_corpus['y'].map(lambda x: 1 if x == 'action' else 0)
    
    sentence_vec, vocab = rnn_tokenizer(comics_corpus['description'], max_len=MAX_LEN)
    X_vec = sentence_vec(comics_corpus['description'])

    MAX_LEN = max([len(text) for text in X_vec])
    
    X_pad = rnn_padding(X_vec, maxlen=MAX_LEN)
    labels = comics_corpus['y'].to_numpy().reshape(-1, 1)
    comics_tokens = np.concatenate([X_pad, labels], axis=1)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    comics_transformers = tokenizer(
        comics_corpus['description'].tolist(),
        return_tensors='pt',
        padding=True
    )

    comics_corpus.to_csv('../data/preprocessed/comics_corpus.csv', index=False)
    np.save('../data/preprocessed/comics_tokens.npy', comics_tokens)
    np.save('../data/preprocessed/comics_transformers', comics_transformers)
