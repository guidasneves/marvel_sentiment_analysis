import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences

import os


def rnn_preprocess(sentence):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')
    
    sentence = sentence.lower()
    sentence = re.sub(r'\W+', ' ', sentence)
    clean_sentence = []
    
    for word in re.split(r'(\W+)', sentence):
        if word not in stopwords_en:
            stem_word = stemmer.stem(word)
            clean_sentence.append(stem_word)

    clean_sentence = ''.join(clean_sentence)
    clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
    
    return clean_sentence


def rnn_vectorization(corpus, max_tokens=None):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    sentence_vec = TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=max_tokens,
        ragged=True
    )
    sentence_vec.adapt(corpus)
    vocabulary = sentence_vec.get_vocabulary(include_special_tokens=True)

    return sentence_vec, vocabulary


def rnn_padding(corpus, maxlen=None, shuffle_buffer_size=1000, prefetch_buffer_size=AUTOTUNE, batch_size=128):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    padded_corpus = pad_sequences(
        corpus.numpy(),
        maxlen=maxlen,
        truncating='post',
        padding='post'
    )

    return padded_corpus


def rnn_batch_dataset(corpus, labels, max_len=None, shuffle_buffer_size=1000, prefetch_buffer_size=AUTOTUNE, batch_size=128):
    """
    [EN-US]
    
    [PT-BR]
    
    """
    dataset = (
        Dataset.from_tensor_slices((corpus, labels))
        .cache()
        .shuffle(shuffle_buffer_size)
        .prefetch(prefetch_buffer_size)
        .batch(batch_size)
    )

    return dataset


if __name__ == '__main__':
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )
    )
    
    comics_data = pd.read_csv(os.path.join(PATH, 'raw', 'comics_corpus.csv'))

    MAX_TOKENS = None
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 128

    comics_data['description'] = comics_data['description'].map(rnn_preprocess)
    comics_data = comics_data.drop_duplicates('description')
    
    sentence_vec, vocab = rnn_vectorization(comics_data['description'], max_tokens=MAX_TOKENS)
    X_vec = sentence_vec(comics_data['description'])
    
    MAX_LEN = max([len(text) for text in X_vec])
    
    X_pad = rnn_padding(X_vec, maxlen=MAX_LEN)
    labels = (comics_data['y'] == 'action').astype(int).to_numpy().reshape(-1, 1)
    
    comics_corpus = np.concatenate([X_pad, labels], axis=1)
    comics_corpus.tofile(os.path.join(PATH, 'preprocessed', 'comics_corpus.csv'))
